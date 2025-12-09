import sys
import os

# FORCE PATH: Add virtual environment site-packages
venv_path = "/home/omrode/Om/Projects/VLA/venv/lib/python3.12/site-packages"
if venv_path not in sys.path:
    sys.path.append(venv_path)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.subscription = self.create_subscription(
            String,
            '/user_command',
            self.command_callback,
            10)
        self.publisher = self.create_publisher(String, '/task_plan', 10)
        self.get_logger().info('LLM Node Ready. Waiting for commands...')

    def command_callback(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f'Received command: {command}')
        
        # Simple Mock Parsing logic
        # "Pick the red cube and place it on the blue platform"
        task = {}
        
        if "pick" in command and "place" in command:
            task['action'] = 'pick_and_place'
            
            # Extract object
            if "red cube" in command:
                task['object'] = 'red_cube'
            elif "green cube" in command:
                task['object'] = 'green_cube'
            elif "yellow cylinder" in command:
                task['object'] = 'yellow_cylinder'
            else:
                task['object'] = 'unknown'
                
            # Extract target
            if "blue platform" in command:
                task['target'] = 'blue_platform'
            else:
                task['target'] = 'unknown'
        else:
            task['action'] = 'unknown'
            
        # Publish
        task_str = json.dumps(task)
        self.publisher.publish(String(data=task_str))
        self.get_logger().info(f'Published Plan: {task_str}')

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

