import sys
import os

# FORCE PATH: Add virtual environment site-packages
venv_path = "/home/omrode/Om/Projects/VLA/venv/lib/python3.12/site-packages"
if venv_path not in sys.path:
    sys.path.append(venv_path)

import rclpy
from rclpy.node import Node
import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray

class VLASimulationNode(Node):
    def __init__(self):
        super().__init__('vla_simulation_node')
        self.get_logger().info('Initializing VLA Simulation Node...')

        self.declare_parameter('use_gui', True)
        self.use_gui = self.get_parameter('use_gui').get_parameter_value().bool_value
        
        # Initialize PyBullet
        connection_mode = p.GUI if self.use_gui else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0) # We will step manually

        # Load Ground
        self.plane_id = p.loadURDF("plane.urdf")

        # Load Robot (Franka Panda)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        
        # Reset robot to a "home" configuration
        joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.0, 0.0]
        for i in range(7):
            p.resetJointState(self.robot_id, i, joint_positions[i])
            
        # Enable high friction for fingers (indices 9 and 10 usually)
        # We need to find them strictly to be sure
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if b'finger' in info[1]:
                p.changeDynamics(self.robot_id, i, lateralFriction=2.0, spinningFriction=0.1)

        # Spawn Objects
        self.spawn_scene()

        # Camera Setup
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.width = 640
        self.height = 480
        
        # Camera position: Fixed above the table looking down/angled
        cam_target_pos = [0.5, 0, 0]
        cam_distance = 1.0
        cam_yaw = 90
        cam_pitch = -40
        cam_roll = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target_pos,
            distance=cam_distance,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=cam_roll,
            upAxisIndex=2
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=100.0
        )
        # Precompute intrinsics from projection
        fov = 60.0
        fx = (self.width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
        fy = fx * (self.height / self.width)
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.cam_info = CameraInfo()
        self.cam_info.height = self.height
        self.cam_info.width = self.width
        self.cam_info.distortion_model = "plumb_bob"
        self.cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.cam_info.k = [fx, 0.0, cx,
                           0.0, fy, cy,
                           0.0, 0.0, 1.0]
        self.cam_info.r = [1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0]
        # Simple pinhole with no offset
        self.cam_info.p = [fx, 0.0, cx, 0.0,
                           0.0, fy, cy, 0.0,
                           0.0, 0.0, 1.0, 0.0]

        # ROS 2 Interface
        self.joint_sub = self.create_subscription(Float64MultiArray, '/joint_commands', self.joint_callback, 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_states', 10)

        # Timers
        # Physics Timer (240Hz)
        self.physics_timer = self.create_timer(1.0 / 240.0, self.step_simulation)
        
        # Camera Timer (10Hz)
        self.camera_timer = self.create_timer(1.0 / 10.0, self.publish_camera_feed)
        # Joint state publisher (30Hz)
        self.joint_state_timer = self.create_timer(1.0 / 30.0, self.publish_joint_states)
        
        self.get_logger().info('VLA Simulation Node Ready.')

    def joint_callback(self, msg):
        target_pos = msg.data
        
        # Panda indices (usually 0-6 are arm, 9-10 are fingers, skipping fixed joints?)
        # Let's dynamically find movable joints if not already done
        if not hasattr(self, 'movable_joints'):
             self.movable_joints = []
             for i in range(p.getNumJoints(self.robot_id)):
                 info = p.getJointInfo(self.robot_id, i)
                 # info[2] is joint type, 4 is fixed
                 if info[2] != p.JOINT_FIXED:
                     self.movable_joints.append(i)
        
        # Apply control
        # We assume input is for 9 joints: 7 arm + 2 fingers
        count = min(len(target_pos), len(self.movable_joints))
        if count > 0:
            # Separate arm and gripper
            forces = [500.0] * count
            
            # TUNE CONTROL: Add max velocity and position gains to prevent oscillation
            # Arm joints (0-6) need smooth motion. Gripper (7-8) needs force.
            # Using position gains (Kp) and velocity gains (Kv/Damping)
            
            p.setJointMotorControlArray(
                self.robot_id, 
                self.movable_joints[:count], 
                p.POSITION_CONTROL, 
                targetPositions=target_pos[:count],
                forces=[60.0] * count,          # further lower force to reduce buzz
                positionGains=[0.06] * count,   # softer
                velocityGains=[1.2] * count     # more damping
            )


    def spawn_scene(self):
        # 1. Blue Platform (Target)
        self.platform_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.02], rgbaColor=[0, 0, 1, 1])
        self.platform_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.02])
        self.platform_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=self.platform_collision_shape,
            baseVisualShapeIndex=self.platform_visual_shape,
            basePosition=[0.5, 0.3, 0.02] 
        )

        # 2. Red Cube (Object to pick)
        self.cube_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[1, 0, 0, 1])
        self.cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        self.cube_id = p.createMultiBody(
            baseMass=0.1, 
            baseCollisionShapeIndex=self.cube_collision_shape,
            baseVisualShapeIndex=self.cube_visual_shape,
            basePosition=[0.5, -0.2, 0.025]
        )
        p.changeDynamics(self.cube_id, -1, lateralFriction=1.0, spinningFriction=0.001, rollingFriction=0.001)

        # 3. Green Cube
        self.green_cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[0, 1, 0, 1])
        self.green_cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        self.green_cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=self.green_cube_collision,
            baseVisualShapeIndex=self.green_cube_visual,
            basePosition=[0.6, -0.2, 0.025]
        )
        p.changeDynamics(self.green_cube_id, -1, lateralFriction=1.0, spinningFriction=0.001, rollingFriction=0.001)

        # 4. Yellow Cylinder
        self.yellow_cyl_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.05, rgbaColor=[1, 1, 0, 1])
        self.yellow_cyl_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.05)
        self.yellow_cyl_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=self.yellow_cyl_collision,
            baseVisualShapeIndex=self.yellow_cyl_visual,
            basePosition=[0.4, -0.2, 0.025]
        )
        p.changeDynamics(self.yellow_cyl_id, -1, lateralFriction=1.0, spinningFriction=0.001, rollingFriction=0.001)


        # 5. White Cup (Cylinder for YOLO Detection)
        self.cup_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.035,
            length=0.09,
            rgbaColor=[1, 1, 1, 1]  # white cup
        )

        self.cup_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.035,
            height=0.09
        )

        self.cup_id = p.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=self.cup_collision_shape,
            baseVisualShapeIndex=self.cup_visual_shape,
            basePosition=[0.45, -0.05, 0.045]
        )

        p.changeDynamics(
            self.cup_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.001,
            rollingFriction=0.001
        )

        
        self.get_logger().info('Scene Spawned: Red/Green Cubes, Yellow Cylinder, Blue Platform, White Cup.')

    def step_simulation(self):
        p.stepSimulation()

    def publish_camera_feed(self):
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER
        images = p.getCameraImage(self.width, self.height, self.view_matrix, self.proj_matrix, renderer=renderer)
        
        # RGB
        rgb_opengl = np.reshape(images[2], (self.height, self.width, 4))
        rgb_image = rgb_opengl[:, :, :3] # Remove alpha
        ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = "camera_link"
        self.image_pub.publish(ros_image)
        # Camera info
        self.cam_info.header = ros_image.header
        self.cam_info_pub.publish(self.cam_info)

        # Depth: OpenGL depth buffer to meters
        depth_buffer = np.reshape(images[3], (self.height, self.width))
        near = 0.1
        far = 100.0
        depth_m = (2.0 * near * far) / (far + near - (2.0 * depth_buffer - 1.0) * (far - near))
        depth_msg = Image()
        depth_msg.header.stamp = ros_image.header.stamp
        depth_msg.header.frame_id = "camera_link"
        depth_msg.height = self.height
        depth_msg.width = self.width
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = False
        depth_msg.step = self.width * 4
        depth_msg.data = depth_m.astype(np.float32).tobytes()
        self.depth_pub.publish(depth_msg)

    def publish_joint_states(self):
        if not hasattr(self, 'movable_joints'):
            return
        states = p.getJointStates(self.robot_id, self.movable_joints)
        positions = [s[0] for s in states]
        msg = Float64MultiArray()
        msg.data = [float(x) for x in positions]
        self.joint_pub.publish(msg)

    def __del__(self):
        try:
            if p.isConnected():
                p.disconnect()
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VLASimulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in simulation loop: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
