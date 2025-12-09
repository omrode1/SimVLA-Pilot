import sys
import os

# FORCE PATH: Add virtual environment site-packages
venv_path = "/home/omrode/Om/Projects/VLA/venv/lib/python3.12/site-packages"
if venv_path not in sys.path:
    sys.path.append(venv_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import json
from std_msgs.msg import String

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.get_logger().info('Initializing Perception Node...')
        
        self.bridge = CvBridge()
        # Load YOLO model (will download on first run)
        self.model = YOLO('yolov8n.pt') 
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        self.detection_pub = self.create_publisher(String, '/perception/detections', 10)
        self.get_logger().info('Perception Node Ready.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        # Run YOLO; use COCO classes and map to known labels
        results = self.model(cv_image, verbose=False)
        
        detections = {}

        # Allowed COCO classes mapped to our labels
        allowed = {
            "cup": "cup",
            "bottle": "bottle",
            "bowl": "bowl",
            "wine glass": "wine_glass",
        }
        conf_thresh = 0.35

        # First pass: YOLO boxes with class labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, "")
                if cls_name not in allowed:
                    continue
                conf = float(box.conf[0].item())
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # clamp
                x1 = max(0, min(x1, cv_image.shape[1] - 1))
                x2 = max(0, min(x2, cv_image.shape[1] - 1))
                y1 = max(0, min(y1, cv_image.shape[0] - 1))
                y2 = max(0, min(y2, cv_image.shape[0] - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                label = allowed[cls_name]
                if label and label not in detections:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections[label] = (cx, cy)

        # Fallback: Color Analysis on full image for any missing labels
        # Red
        lower_red = np.array([0, 0, 100])
        upper_red = np.array([50, 50, 255])
        mask_red = cv2.inRange(cv_image, lower_red, upper_red)
        
        # Green
        lower_green = np.array([0, 100, 0])
        upper_green = np.array([50, 255, 50])
        mask_green = cv2.inRange(cv_image, lower_green, upper_green)

        # Yellow (Red + Green)
        lower_yellow = np.array([0, 100, 100])
        upper_yellow = np.array([50, 255, 255])
        mask_yellow = cv2.inRange(cv_image, lower_yellow, upper_yellow)
        
        # Blue
        lower_blue = np.array([100, 0, 0])
        upper_blue = np.array([255, 50, 50])
        mask_blue = cv2.inRange(cv_image, lower_blue, upper_blue)
        
        # Find centroids of color blobs
        def get_centroid(mask):
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return cX, cY
            return None
            
        red_center = get_centroid(mask_red)
        green_center = get_centroid(mask_green)
        yellow_center = get_centroid(mask_yellow)
        blue_center = get_centroid(mask_blue)
        
        if red_center and 'red_cube' not in detections:
            detections['red_cube'] = red_center
        if green_center and 'green_cube' not in detections:
            detections['green_cube'] = green_center
        if yellow_center and 'yellow_cylinder' not in detections:
            detections['yellow_cylinder'] = yellow_center
        if blue_center and 'blue_platform' not in detections:
            detections['blue_platform'] = blue_center
            
        # Publish JSON
        if detections:
            msg_str = json.dumps(detections)
            self.detection_pub.publish(String(data=msg_str))
            # self.get_logger().info(f'Published: {msg_str}')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

