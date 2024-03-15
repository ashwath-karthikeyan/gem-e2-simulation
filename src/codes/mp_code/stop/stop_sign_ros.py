#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)

        # Set up subscribers and publishers
        self.image_sub = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.image_callback)
        # self.image_pub = rospy.Publisher('/zed2/zed_node/rgb/image_processed', Image, queue_size=10)
        self.stopsign_pub = rospy.Publisher('/zed2/zed_node/rgb/stopsign', Image, queue_size=10)

        self.bridge = CvBridge()

    def stop_sign(self, img):

        stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

        # img = cv2.imread('stop_landscape.jpeg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

        # Detect the stop sign, x,y = origin points, w = width, h = height
        for (x, y, w, h) in stop_sign_scaled:
            # Draw rectangle around the stop sign
            stop_sign_rectangle = cv2.rectangle(img, (x, y),
                                                (x + w, y + h),
                                                (0, 255, 0), 3)
            # Write "Stop sign" on the bottom of the rectangle
            stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                        text="",
                                        org=(x, y + h + 30),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(0, 0, 255),
                                        thickness=2, lineType=cv2.LINE_4)

        return img
        

    def image_callback(self, msg):

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Perform image manipulations (e.g., resizing, color conversion, etc.)
            # Modify the image processing steps according to your requirements

            processed_image = self.stop_sign(cv_image);

            # Convert the processed image back to ROS Image message
            stopsign_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')

            # Publish the processed image to a new topic
            self.stopsign_pub.publish(stopsign_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

            
if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass