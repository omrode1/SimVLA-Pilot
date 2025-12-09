import sys
import os

# FORCE PATH: Add virtual environment site-packages
venv_path = "/home/omrode/Om/Projects/VLA/venv/lib/python3.12/site-packages"
if venv_path not in sys.path:
    sys.path.append(venv_path)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import Image
import json
import time
import pybullet as p
import pybullet_data
import numpy as np
from collections import deque

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.get_logger().info('Initializing Planner Node...')
        
        # Subs/Pubs
        self.task_sub = self.create_subscription(String, '/task_plan', self.task_callback, 10)
        self.perception_sub = self.create_subscription(String, '/perception/detections', self.perception_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.joint_state_sub = self.create_subscription(Float64MultiArray, '/joint_states', self.joint_state_callback, 10)
        
        # State
        self.state = "IDLE"
        self.current_task = None
        self.object_pos = None
        self.target_pos = None
        self.state_start = 0.0
        self.current_joints = None
        self.detect_ema = {}
        self.depth_image = None
        self.cmd_window = deque(maxlen=5)
        self.last_state = None
        
        # IK Solver Setup (Headless PyBullet)
        self.ik_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        # Set home
        home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.0, 0.0]
        for i in range(7):
            p.resetJointState(self.robot_id, i, home_joints[i])
            
        # Camera intrinsics/extrinsics (match simulation_node)
        self.cam_width = 640
        self.cam_height = 480
        self.cam_target = [0.5, 0, 0]
        self.cam_distance = 1.0
        self.cam_yaw = 90
        self.cam_pitch = -40
        self.cam_roll = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target,
            distance=self.cam_distance,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=self.cam_roll,
            upAxisIndex=2
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.cam_width / self.cam_height,
            nearVal=0.1,
            farVal=100.0
        )
        vm = np.array(self.view_matrix).reshape(4, 4).T
        pm = np.array(self.proj_matrix).reshape(4, 4).T
        self.inv_viewproj = np.linalg.inv(pm @ vm)
        self.inv_view = np.linalg.inv(vm)
        self.cam_pos = self.inv_view[:3, 3]
        
        # Run faster so we can smooth trajectories (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        self.prev_cmd = None
        
        self.get_logger().info('Planner Node Ready.')

    def task_callback(self, msg):
        try:
            task = json.loads(msg.data)
            self.get_logger().info(f"Received Task: {task}")
            if task.get('action') == 'pick_and_place':
                self.current_task = task
                self.state = "LOCATING"
        except Exception as e:
            self.get_logger().error(f"Failed to parse task: {e}")

    def joint_state_callback(self, msg):
        # Store latest measured joints from simulation
        self.current_joints = list(msg.data)

    def get_depth_at(self, u, v, k=1):
        if self.depth_image is None:
            return None
        u = int(u)
        v = int(v)
        us = max(0, u - k)
        ue = min(self.depth_image.shape[1], u + k + 1)
        vs = max(0, v - k)
        ve = min(self.depth_image.shape[0], v + k + 1)
        patch = self.depth_image[vs:ve, us:ue]
        if patch.size == 0:
            return None
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def depth_callback(self, msg):
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            self.depth_image = depth
        except Exception as e:
            self.get_logger().error(f"Depth parse error: {e}")

    def pixel_to_world(self, u, v, plane_z=0.025, depth_m=None):
        # Convert pixel (u,v) to world coordinates using depth if available; else intersect plane_z
        ndc_x = (2.0 * u) / self.cam_width - 1.0
        ndc_y = 1.0 - (2.0 * v) / self.cam_height
        near = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far = np.array([ndc_x, ndc_y, 1.0, 1.0])
        near_world = self.inv_viewproj @ near
        far_world = self.inv_viewproj @ far
        near_world /= near_world[3]
        far_world /= far_world[3]
        ray_dir = far_world[:3] - near_world[:3]
        ray_dir_norm = np.linalg.norm(ray_dir)
        if ray_dir_norm < 1e-8:
            return None
        ray_dir = ray_dir / ray_dir_norm

        if depth_m is not None and depth_m > 0:
            point = self.cam_pos + ray_dir * depth_m
            return [float(point[0]), float(point[1]), float(point[2])]

        if abs(ray_dir[2]) < 1e-6:
            return None
        t = (plane_z - near_world[2]) / ray_dir[2]
        point = near_world[:3] + t * ray_dir
        return [float(point[0]), float(point[1]), float(plane_z)]

    def perception_callback(self, msg):
        try:
            detections = json.loads(msg.data)
            # Smooth pixel detections (EMA) to reduce jitter
            ema_alpha = 0.2
            smoothed = {}
            for name, pix in detections.items():
                if name in self.detect_ema:
                    prev = self.detect_ema[name]
                    smoothed[name] = [
                        prev[0] + ema_alpha * (pix[0] - prev[0]),
                        prev[1] + ema_alpha * (pix[1] - prev[1])
                    ]
                else:
                    smoothed[name] = pix
                self.detect_ema[name] = smoothed[name]
            detections = smoothed
            
            # Verification Logic
            if self.state == "VERIFYING" and self.current_task:
                obj_name = self.current_task.get('object')
                tgt_name = self.current_task.get('target')
                
                if obj_name in detections and tgt_name in detections:
                    obj_pix = detections[obj_name]
                    tgt_pix = detections[tgt_name]
                    
                    dist = np.linalg.norm(np.array(obj_pix) - np.array(tgt_pix))
                    self.get_logger().info(f"Verification Dist: {dist:.2f} pixels")
                    
                    if dist < 100: # Threshold (pixels)
                        self.get_logger().info("SUCCESS: Object placed on target!")
                    else:
                        self.get_logger().warn("FAILURE: Object not on target.")
                    
                    self.state = "IDLE"
                    return

            # Locating Logic
            if self.state == "LOCATING" and self.current_task:
                obj_name = self.current_task.get('object')
                tgt_name = self.current_task.get('target')
                
                if obj_name in detections:
                    obj_pix = detections[obj_name]
                    # Use depth to get top of object, then calculate pick point below it
                    depth_m = self.get_depth_at(obj_pix[0], obj_pix[1], k=2)
                    obj_world = self.pixel_to_world(obj_pix[0], obj_pix[1], plane_z=0.025, depth_m=None)
                    if obj_world:
                        # Calculate pick Z: table plane (0.025) minus clearance to go below object
                        # This ensures gripper goes below object surface, not just touching top
                        pick_z = max(0.005, 0.025 - 0.020)  # ~0.5 cm above ground, well below cube
                        self.object_pos = [obj_world[0], obj_world[1], pick_z]
                        self.get_logger().info(f"Pick Z calculated: {pick_z:.4f} m (table: 0.025m, clearance: 0.020m)")
                    
                if tgt_name in detections:
                    tgt_pix = detections[tgt_name]
                    depth_m = self.get_depth_at(tgt_pix[0], tgt_pix[1], k=2)
                    tgt_world = self.pixel_to_world(tgt_pix[0], tgt_pix[1], plane_z=0.02, depth_m=depth_m)
                    if tgt_world:
                        self.target_pos = tgt_world
                    
                    if self.object_pos and self.target_pos:
                        self.get_logger().info(f"Objects Located: {obj_name} -> {self.object_pos} (pick Z forced to table: 0.025m)")
                        self.state = "MOVING_TO_PICK"
                        self.state_start = time.time()
        except Exception as e:
            self.get_logger().error(f"Perception error: {e}")

    def calculate_ik(self, target_pos, target_orn=None):
        if target_orn is None:
             # Point gripper down
             target_orn = p.getQuaternionFromEuler([3.14159, 0, 0]) 
        
        # Sync IK model with current sim joints to reduce IK jumps
        if self.current_joints is not None:
            for i in range(min(7, len(self.current_joints))):
                p.resetJointState(self.robot_id, i, self.current_joints[i])

        # Link 11 is tip
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, 
            11, 
            target_pos, 
            target_orn,
            residualThreshold=0.001,
            maxNumIterations=100
        )
        # Result contains all movable joints (9)
        # We need first 7 for arm
        return list(joint_poses)[:7]

    def control_loop(self):
        if self.state == "IDLE" or self.state == "LOCATING":
            return

        # Reset smoothing on state change to avoid dragging old gripper targets
        if self.last_state != self.state:
            self.prev_cmd = None
            self.cmd_window.clear()
            self.last_state = self.state

        cmd = []
        # Timed State Machine
        now = time.time()
        elapsed = now - self.state_start
        
        if self.state == "MOVING_TO_PICK":
            # Move above object
            hover_pos = [self.object_pos[0], self.object_pos[1], 0.25]
            joints = self.calculate_ik(hover_pos)
            cmd = joints + [0.04, 0.04] # Open
            
            if elapsed > 2.0:
                 self.state = "DESCENDING_PICK"
                 self.state_start = now
                 self.get_logger().info("Descending to Pick")

        elif self.state == "DESCENDING_PICK":
            # Keep gripper open while descending; use precomputed pick Z
            pick_pos = [self.object_pos[0], self.object_pos[1], self.object_pos[2]]
            joints = self.calculate_ik(pick_pos)
            cmd = joints + [0.04, 0.04]
            
            if elapsed > 2.0:
                 self.state = "GRASPING"
                 self.state_start = now
                 self.get_logger().info("Grasping")

        elif self.state == "GRASPING":
            # Use the same pick Z position
            pick_pos = [self.object_pos[0], self.object_pos[1], self.object_pos[2]]
            joints = self.calculate_ik(pick_pos)
            # Close tightly (slightly negative tolerance for squeeze)
            cmd = joints + [-0.005, -0.005] 
            
            if elapsed > 3.0:
                 self.state = "LIFTING"
                 self.state_start = now
                 self.get_logger().info("Lifting")

        elif self.state == "LIFTING":
             hover_pos = [self.object_pos[0], self.object_pos[1], 0.25]
             joints = self.calculate_ik(hover_pos)
             # Keep closed
             cmd = joints + [0.01, 0.01]
             
             if elapsed > 2.0:
                 self.state = "MOVING_TO_PLACE"
                 self.state_start = now
                 self.get_logger().info("Moving to Place")

        elif self.state == "MOVING_TO_PLACE":
             hover_pos = [self.target_pos[0], self.target_pos[1], 0.25]
             joints = self.calculate_ik(hover_pos)
             cmd = joints + [0.01, 0.01]
             
             if elapsed > 3.0:
                 self.state = "DESCENDING_PLACE"
                 self.state_start = now
                 self.get_logger().info("Descending to Place")

        elif self.state == "DESCENDING_PLACE":
             place_pos = [self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.04]
             joints = self.calculate_ik(place_pos)
             cmd = joints + [0.01, 0.01]
             
             if elapsed > 2.0:
                 self.state = "RELEASING"
                 self.state_start = now
                 self.get_logger().info("Releasing")

        elif self.state == "RELEASING":
             place_pos = [self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.04]
             joints = self.calculate_ik(place_pos)
             cmd = joints + [0.05, 0.05] # Open
             
             if elapsed > 1.0:
                 self.state = "HOME"
                 self.state_start = now
                 self.get_logger().info("Going Home")
                 
        elif self.state == "HOME":
             home_pos = [0.5, 0, 0.5]
             joints = self.calculate_ik(home_pos)
             cmd = joints + [0.04, 0.04]
             
             if elapsed > 3.0:
                 self.get_logger().info("Task Completed. Verifying...")
                 self.state = "VERIFYING"
                 self.state_start = now

        if cmd:
            self.publish_joints(cmd)

    def publish_joints(self, joints):
        msg = Float64MultiArray()
        # Smoothing + rate limiting to avoid jerky jumps
        if self.prev_cmd is None:
            self.prev_cmd = list(joints)
        # Slower increment to reduce oscillation
        alpha = 0.015
        max_delta = 0.004  # rad per step
        deadband = 0.006  # if close enough, hold
        smooth = []
        for p_prev, p_target in zip(self.prev_cmd, joints):
            error = p_target - p_prev
            if abs(error) < deadband:
                interp = p_prev
            else:
                interp = p_prev + alpha * error
            delta = interp - p_prev
            if delta > max_delta:
                interp = p_prev + max_delta
            elif delta < -max_delta:
                interp = p_prev - max_delta
            smooth.append(interp)
        self.prev_cmd = smooth

        # Moving average over recent commands to further smooth
        self.cmd_window.append(smooth)
        if len(self.cmd_window) > 1:
            avg = [sum(vals) / len(self.cmd_window) for vals in zip(*self.cmd_window)]
        else:
            avg = smooth

        msg.data = [float(x) for x in avg]
        self.joint_pub.publish(msg)
        
    def __del__(self):
        try:
            p.disconnect(self.ik_client)
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

