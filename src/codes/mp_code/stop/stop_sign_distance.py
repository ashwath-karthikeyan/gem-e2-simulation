#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)

        # Subscribers and publishers
        self.image_sub = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.image_callback)
        self.stopsign_pub = rospy.Publisher('/zed2/zed_node/rgb/stopsign', Image, queue_size=10)

        self.bridge = CvBridge()

        # Constants for distance estimation (set these based on your calibration)
        self.S0 = 4.7
        self.D0 = 247 # known distance corresponding to S0

    def stop_sign(self, img):
        stop_sign = cv2.CascadeClassifier('cascade_stop_sign_new.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in stop_sign_scaled:
            # Draw rectangle around the stop sign
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Calculate current size (using width or height)
            current_size = w  # or h, depending on your setup

            # Estimate distance
            current_distance = (self.S0 * self.D0 / current_size) - 6

            # Display the estimated distance
            cv2.putText(img, f"Distance: {current_distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            processed_image = self.stop_sign(cv_image)
            stopsign_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')
            self.stopsign_pub.publish(stopsign_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass