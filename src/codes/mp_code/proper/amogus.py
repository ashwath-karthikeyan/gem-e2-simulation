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
        self.image_pub = rospy.Publisher('/zed2/zed_node/rgb/image_processed', Image, queue_size=10)
        self.overlay_pub = rospy.Publisher('/zed2/zed_node/rgb/overlay', Image, queue_size=10)
        self.depth_sub = rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, self.depth_callback)        

        self.depth = None
        self.middle_curve = None # grayscale
        self.uniform_midpoint_distance = None
        self.bridge = CvBridge()

    def process_image(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        #darkened = cv2.addWeighted(gray, 1.5, np.zeros_like(gray), 0, 0)

        # Isolate white from HLS to get white mask
        # lower_white = np.array([0, 80, 0], dtype=np.uint8)
        # upper_white = np.array([255, 255, 255], dtype=np.uint8)

        lower_white = np.array([0, 160, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)


        white_mask = cv2.inRange(hls, lower_white, upper_white)

        masked_image = cv2.bitwise_and(gray, gray, mask=white_mask)

        _, binary_mask = cv2.threshold(masked_image, 120, 255, cv2.THRESH_BINARY)


        sigma = 0.33
        med = np.median(gray)
        lower_thresh = int(max(0, (1.0-sigma) * med))
        upper_tresh = int(min(255, (1.0+sigma) * med))

        # Apply slight Gaussian Blur
        blurred = cv2.GaussianBlur(binary_mask, (15,15), 0)

        # Apply Canny Edge Detector
        edges = cv2.Canny(blurred, lower_thresh, upper_tresh)

        # Create a mask for the top half of the image
        height, width = edges.shape[:2]
        mask = np.zeros_like(edges, dtype=np.uint8)
        mask[(height // 2):, :] = 255  # Set the top half to 255 (white)

        # Apply the mask to erase the top half of the edges image
        # erased_edges_image = cv2.bitwise_and(edges, mask)


        height, width = edges.shape


        middle_len = 0.4
        roi_vertices = np.array([[(width * 0.1, height),      # Bottom-left vertex
                          (width * (0.5 - (middle_len/2)), height * 0.6),       # Top-left vertex
                          (width * (0.5 + (middle_len/2)), height * 0.6),       # Top-right vertex
                          (width * 0.9, height)]],            # Bottom-right vertex
                        dtype=np.int32)

        roi_edges = cv2.fillPoly(np.zeros_like(edges), roi_vertices, 255)
        roi_image = cv2.bitwise_and(edges, roi_edges)

        kernel = np.ones((15, 15), np.uint8)
        roi_image_dilated = cv2.dilate(roi_image, kernel, iterations=1)

        # lines = cv2.HoughLines(roi_image_dilated, 1, np.pi / 180, threshold=200)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_image_dilated, connectivity=8)

        # Set a threshold for the minimum size of connected components (adjust as needed)
        min_component_size = 4000

        # Filter connected components based on size
        filtered_indices = [index for index, size in enumerate(stats[1:, -1], start=1) if size > min_component_size]

        # Take the top two components based on size
        top_two_indices = sorted(filtered_indices, key=lambda index: stats[index, -1], reverse=True)[:2]

        # Create a binary mask for the top two components
        top_two_mask = np.zeros_like(labels, dtype=np.uint8)
        for i, index in enumerate(top_two_indices, start=2):  # Start enumeration from 2
            top_two_mask[labels == index] = i
        
        
        # Create a colored output image
        colored_image = cv2.applyColorMap(top_two_mask * 30, cv2.COLORMAP_JET)

        # Create separate masks for the top two components
        mask1 = np.zeros_like(labels, dtype=np.uint8)
        mask2 = np.zeros_like(labels, dtype=np.uint8)

        for i, index in enumerate(top_two_indices, start=2):
            if i == 2:
                mask1[labels == index] = 255
            elif i == 3:
                mask2[labels == index] = 255

        # Extract the components using bitwise AND
        # component1 = cv2.bitwise_and(roi_image_dilated, roi_image_dilated, mask=mask1)
        # component2 = cv2.bitwise_and(roi_image_dilated, roi_image_dilated, mask=mask2)

        # # Optionally, you can dilate the components for better visualization
        # kernel = np.ones((15, 15), np.uint8)
        # component1 = cv2.dilate(component1, kernel, iterations=1)
        # component2 = cv2.dilate(component2, kernel, iterations=1)
        _, filterd_roi_dilated = cv2.threshold(cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(filterd_roi_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_image = np.zeros_like(image) # contains circles and 3 curves (left, middle right)
        middle_contour = np.zeros_like(image) # contains middle-curve only for future use

        # Draw contours on the empty image
        # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

        contour_points_left = None
        contour_points_right = None
        contour_index = 0

        # Draw representative points along contours and fit lines
        for contour in contours:
            # Simplify the contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

            # Draw circles at the representative points
            # for point in simplified_contour:
            #     x, y = point[0]
            #     # Different intensities to distinguish different contours
            #     if (contour_index == 0):
            #         cv2.circle(contour_image, (x, y), radius=10, color=(100, 100, 100), thickness=-1)
            #     if (contour_index == 1):
            #         cv2.circle(contour_image, (x, y), radius=10, color=(255, 255, 255), thickness=-1)
            # Fit a line through the simplified contour
            
            if len(simplified_contour) > 1:
               
                x = simplified_contour[:, 0, 0]
                y = simplified_contour[:, 0, 1]
                # draw_points = (np.asarray([x, y]).T).astype(np.int32)   # needs to be int32 and transposed
                
                degree_of_polynomial = 2  # Adjust the degree as needed
                coefficients = np.polyfit(y, x, degree_of_polynomial)

                # Create a y-axis range for plotting the polynomial line
                y_range = np.linspace(min(y), max(y), 100)

                # Compute corresponding x values using the polynomial coefficients
                x_values = np.polyval(coefficients, y_range)

                if (contour_index == 1):
                    contour_points_left = np.column_stack((x_values, y_range))
                elif (contour_index == 0):
                    contour_points_right = np.column_stack((x_values, y_range))
                contour_index += 1

                # ~~~~~~ Display of the left and right curves ~~~~~~~~~
                # Draw the polynomial line on the empty image
                for i in range(len(x_values) - 1):
                    x1, y1 = int(x_values[i]), int(y_range[i])
                    x2, y2 = int(x_values[i + 1]), int(y_range[i + 1])
                    # cv2.line(roi_image_dilated, (x1, y1), (x2, y2), (150, 150, 150), 2)

        x_values_middle, y_range_middle = self.generate_waypoints(contour_points_left, contour_points_right)
        # ~~~~~~ Display of the middle-curve ~~~~~~~~~
        for i in range(len(x_values_middle) - 1):
            x1_middle, y1_middle = int(x_values_middle[i]), int(y_range_middle[i])
            x2_middle, y2_middle = int(x_values_middle[i + 1]), int(y_range_middle[i + 1])
            cv2.line(filterd_roi_dilated, (x1_middle, y1_middle), (x2_middle, y2_middle), (255, 255, 255), 3)
            cv2.line(middle_contour, (x1_middle, y1_middle), (x2_middle, y2_middle), (255, 255, 255), 3)

        self.middle_curve = cv2.cvtColor(middle_contour, cv2.COLOR_BGR2GRAY)

        for i in range(len(x_values_middle) - 1):
            x1_middle, y1_middle = int(x_values_middle[i]), int(y_range_middle[i])
            x2_middle, y2_middle = int(x_values_middle[i + 1]), int(y_range_middle[i + 1])
            # Drawing the line in green on the filtered image
            cv2.line(filterd_roi_dilated, (x1_middle, y1_middle), (x2_middle, y2_middle), (0, 255, 0), 3)
            # Drawing the line in green on the original image
            cv2.line(image, (x1_middle, y1_middle), (x2_middle, y2_middle), (0, 255, 0), 3)

        # cv2.imshow("color", colored_image)
        # cv2.waitKey(1)
        return image
        return filterd_roi_dilated # cv2.cvtColor(roi_image_dilated, cv2.COLOR_BGR2GRAY)
        
  
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Perform image manipulations (e.g., resizing, color conversion, etc.)
            # Modify the image processing steps according to your requirements
        
            processed_image = self.process_image(cv_image);

            overlay_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')


            # Convert the processed image back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')

            # Publish the processed image to a new topic
            self.image_pub.publish(processed_msg)
            self.overlay_pub.publish(overlay_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))


    def depth_callback(self, msg):
        try:
            depth_array = np.frombuffer(msg.data, dtype=np.float32)
            depth_image = depth_array.reshape((msg.height, msg.width))
            if (self.middle_curve is not None):
                self.depth = self.filter_depth_matrix(depth_image, self.middle_curve)            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

    def filter_depth_matrix(self, depth_image, mask):
        roi_depth = np.zeros_like(depth_image)
        roi_depth = np.where((mask > 0), depth_image, 0)
        return roi_depth

    def generate_waypoints(self, contour_points_left, contour_points_right):
        if contour_points_left is None and contour_points_right is None:
            print("Missing both lines")
            return np.array([]), np.array([])

        # Initialize an empty list for the middle points
        contour_points_middle = []

        # Calculate the uniform midpoint distance when both lines are visible
        if contour_points_left is not None and contour_points_right is not None:
            print("both lines visible yayy")
            distances = [((x_right - x_left) / 2) for (x_left, _), (x_right, _) in zip(contour_points_left, contour_points_right)]
            self.uniform_midpoint_distance = np.mean(distances)
            print(self.uniform_midpoint_distance)

        # Handle the case where only one line is detected
        if contour_points_left is None or contour_points_right is None:
            # Check if midpoint distance is not yet calculated
            print("only one line visible")
            if self.uniform_midpoint_distance is None:
                print("Uniform midpoint distance is not yet calculated")
                return np.array([]), np.array([])

            single_line = contour_points_left if contour_points_left is not None else contour_points_right
            for x, y in single_line:
                if contour_points_left is None:
                    # Assuming right line is detected, calculate middle points to the left
                    contour_points_middle.append((x + self.uniform_midpoint_distance, y))
                else:
                    # Assuming left line is detected, calculate middle points to the right
                    contour_points_middle.append((x - self.uniform_midpoint_distance, y))
        else:
            # If both lines are detected, calculate middle points as usual
            for (x_left, y_left), (x_right, y_right) in zip(contour_points_left, contour_points_right):
                contour_points_middle.append(((x_left + x_right) / 2, (y_left + y_right) / 2))

        # Convert the list of tuples to a NumPy array
        contour_points_middle = np.array(contour_points_middle, dtype=float)
        x_middle = contour_points_middle[:, 0]
        y_middle = contour_points_middle[:, 1]

        # Fit polynomial to middle curve
        degree_of_polynomial = 2
        coefficients_middle = np.polyfit(y_middle, x_middle, degree_of_polynomial)

        return x_middle, y_middle


            
if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass