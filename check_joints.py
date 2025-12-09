import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("franka_panda/panda.urdf")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    if info[2] != p.JOINT_FIXED:
        print(f"Index {i}: {info[1]} (Movable)")
    else:
        print(f"Index {i}: {info[1]} (Fixed)")
