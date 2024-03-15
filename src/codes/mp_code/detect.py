#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo, NavSatFix
from novatel_gps_msgs.msg import Inspva # for GPS
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os


class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)

        # Set up subscribers and publishers
        self.image_sub = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/zed2/zed_node/rgb/image_processed', Image, queue_size=10)
        self.depth_sub = rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/zed2/zed_node/rgb/camera_info', CameraInfo, self.camera_info_callback)
        self.gps_sub = rospy.Subscriber('/novatel/inspva', Inspva, self.gps_callback)

        self.depth = None
        self.middle_curve = None # grayscale
        self.camera_info = None # = (camera_matrix, distortion_coeffs)
        self.gps = None # = (latitude, longitude, heading)
        self.waypoint = None
        self.bridge = CvBridge()

    def process_image(self, image): 
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        #darkened = cv2.addWeighted(gray, 1.5, np.zeros_like(gray), 0, 0)

        # Isolate white from HLS to get white mask
        # lower_white = np.array([0, 80, 0], dtype=np.uint8)
        # upper_white = np.array([255, 255, 255], dtype=np.uint8)

        lower_white = np.array([0, 20, 0], dtype=np.uint8)
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

        x_values_middle, y_range_middle = self.generate_middle_curve(contour_points_left, contour_points_right)
        # ~~~~~~ Display of the middle-curve ~~~~~~~~~
        for i in range(len(x_values_middle) - 1):
            x1_middle, y1_middle = int(x_values_middle[i]), int(y_range_middle[i])
            x2_middle, y2_middle = int(x_values_middle[i + 1]), int(y_range_middle[i + 1])
            cv2.line(filterd_roi_dilated, (x1_middle, y1_middle), (x2_middle, y2_middle), (255, 255, 255), 3)
            cv2.line(middle_contour, (x1_middle, y1_middle), (x2_middle, y2_middle), (255, 255, 255), 3)

        self.middle_curve = cv2.cvtColor(middle_contour, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("color", colored_image)
        # cv2.waitKey(1)
        
        return filterd_roi_dilated # cv2.cvtColor(roi_image_dilated, cv2.COLOR_BGR2GRAY)
        
  
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Perform image manipulations (e.g., resizing, color conversion, etc.)
            # Modify the image processing steps according to your requirements
        
            processed_image = self.process_image(cv_image); 

            # Convert the processed image back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, 'mono8')

            # Publish the processed image to a new topic
            self.image_pub.publish(processed_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))


    def depth_callback(self, msg):
        try:
            depth_array = np.frombuffer(msg.data, dtype=np.float32)
            depth_image = depth_array.reshape((msg.height, msg.width))
            # if (self.middle_curve is not None):
            #     self.depth = self.filter_depth_matrix(depth_image, self.middle_curve)
            #     if (self.camera_info is not None):
            #         self.waypoints = self.generate_waypoints()            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

    def filter_depth_matrix(self, depth_image, mask):
        roi_depth = np.zeros_like(depth_image)
        roi_depth = np.where((mask > 0), depth_image, 0)
        return roi_depth

    def generate_middle_curve(self, contour_points_left, contour_points_right):
        if (contour_points_left is None and contour_points_right is None):
            print("missing both lines")
            return [], []
        elif (contour_points_left is None):
            print("missing left lane")
            # Set the shift amount to your desired value
            shift_amount = 300  # Adjust as needed

            # Simulate the middle lane by shifting the detected lane (e.g., right lane)
            contour_points_middle = np.zeros_like(contour_points_right)
            for i in range(len(contour_points_right)):
                x_right, y_right = contour_points_right[i]
                x_middle = x_right - shift_amount
                contour_points_middle[i] = (x_middle, y_right)

            # polyfit the middle curve
            x_middle = np.array([point[0] for point in contour_points_middle])
            y_middle = np.array([point[1] for point in contour_points_middle])
            degree_of_polynomial = 2
            coefficients_middle = np.polyfit(y_middle, x_middle, degree_of_polynomial)
            return x_middle, y_middle

        elif (contour_points_right is None): 
            print("missing right lane")
            # Set the shift amount to your desired value
            shift_amount = 300  # Adjust as needed
            # Simulate the middle lane by shifting the detected lane (e.g., right lane)
            contour_points_middle = np.zeros_like(contour_points_left)
            for i in range(len(contour_points_left)):
                x_right, y_right = contour_points_left[i]
                x_middle = x_right + shift_amount
                contour_points_middle[i] = (x_middle, y_right)

            # polyfit the middle curve
            x_middle = np.array([point[0] for point in contour_points_middle])
            y_middle = np.array([point[1] for point in contour_points_middle])
            degree_of_polynomial = 2
            coefficients_middle = np.polyfit(y_middle, x_middle, degree_of_polynomial)
            return x_middle, y_middle

        else:
            # size = 0
            # if (len(contour_points_left) <= len(contour_points_right)):
            #     size = len(contour_points_left)
            # else:
            #     size = len(contour_points_right)
            #contour_points_middle = np.empty((size,), dtype=np.ndarray)
            contour_points_middle = np.zeros_like(contour_points_left)
            # contour_points_left = sorted(contour_points_left, key=lambda point: point[0])
            # contour_points_right = sorted(contour_points_right, key=lambda point: point[0])

            # get 100 points from the middle curve (between the left and right curve)
            for i in range(len(contour_points_left)):
                x_left, y_left = contour_points_left[i]
                x_right, y_right = contour_points_right[i]

                x_middle = (x_left + x_right)/2
                y_middle = (y_left + y_right)/2

                contour_points_middle[i] = (x_middle, y_middle)

            # polyfit the middle curve
            x_middle = np.array([point[0] for point in contour_points_middle])
            y_middle = np.array([point[1] for point in contour_points_middle])
            degree_of_polynomial = 2 
            coefficients_middle = np.polyfit(y_middle, x_middle, degree_of_polynomial)

            print("Have both lanes")

            return x_middle, y_middle


    def camera_info_callback(self, msg):
        # Access camera matrix and distortion coefficients
        camera_matrix = np.array(msg.K).reshape((3, 3))
        distortion_coeffs = msg.D
        self.camera_info = (camera_matrix, distortion_coeffs)

    def pixel_to_3D(self, camera_matrix, distortion_coeffs, pixel_coordinates):

        pixel_coordinates = pixel_coordinates.reshape((len(pixel_coordinates), 1, 2))

        # Undistort the pixel coordinates
        undistorted_coordinates = cv2.undistortPoints(pixel_coordinates, camera_matrix, distortion_coeffs)

        # Project to 3D real-world coordinates
        num_points = undistorted_coordinates.shape[0]
        homogeneous_coordinates = np.zeros((num_points, 3))

        for i in range(num_points):
            homogeneous_coordinates[i] = np.hstack((undistorted_coordinates[i][0], np.ones(1)))

        # Invert the camera matrix
        inverse_camera_matrix = np.linalg.inv(camera_matrix)

        # Transform homogeneous coordinates to world coordinates
        world_coordinates_homogeneous = inverse_camera_matrix.dot(homogeneous_coordinates.T).T

        # Extract 3D real-world coordinates
        world_coordinates = world_coordinates_homogeneous / world_coordinates_homogeneous[:, -1:]
        #print("finished pixel_to_3D")

        return world_coordinates


    def generate_waypoints(self):

        # Obtain an array of all the (x, y) where depth_matrix[y][x] is larger than 0
        depth_matrix = self.depth
        nonzero_indices = np.where(depth_matrix > 0)
        coordinates_list = list(zip(nonzero_indices[1], nonzero_indices[0]))
        coordinates_array = np.array(coordinates_list, dtype=np.float32)

        # Extract the values from the depth_matrix at the non-zero indices
        middle_curve_depth = depth_matrix[nonzero_indices]

        # Get xy in pixel frame for the middle curve
        #pixel_coordinates = np.array([[0,0], [1200,560]], dtype=np.float32)
        pixel_coordinates = coordinates_array

        # Get XYZ coordinates
        world_coordinates = self.pixel_to_3D(self.camera_info[0], self.camera_info[1], pixel_coordinates)
        # world_coordinates.shape = (num_points, 3)

        # Combine XYZ with Depth
        XYZD_middle_curve = np.hstack((world_coordinates, middle_curve_depth.reshape((-1, 1))))
        # XYZD_middle_curve.shape = (num_points, 4)

        # waypoints = [(X, Y, heading), ...]
        waypoints = np.zeros((world_coordinates.shape[0], 3))

        # Convert to actual lat, long, and heading
        if (self.gps is not None):
            # From https://en.wikipedia.org/wiki/Rotation_of_axes_in_two_dimensions
            theta = -self.gps[2]
            rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                        [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
            waypoints[:, :2] = XYZD_middle_curve[:, :2]
            waypoints[:, :2] = np.dot(waypoints[:, :2], rotation_matrix.T)
            waypoints[:, :2] = waypoints[:, :2] + np.array(self.gps[:2])



            sampling_interval = 20       # <- Average groups of every __ waypoints
            sampled_waypoints = np.zeros((waypoints.shape[0] // sampling_interval, 3))
            sampled_waypoints = np.array([
                    [np.mean(waypoints[i * sampling_interval:(i + 1) * sampling_interval, 0]),
                    np.mean(waypoints[i * sampling_interval:(i + 1) * sampling_interval, 1]),
                    waypoints[0, 2]]
                    for i in range(waypoints.shape[0] // sampling_interval)
                ])

            # Find relative heading
            diffs = sampled_waypoints[:, :2] - sampled_waypoints[0, :2]  # Calculate differences between each waypoint and the first waypoint
            relative_heading = np.arctan2(diffs[:, 1], diffs[:, 0])  # Calculate the angle using arctan2
            relative_heading = np.degrees(relative_heading)  # Convert angles to degrees if needed

            gps_heading = relative_heading - self.gps[2]
            sampled_waypoints[:, 2] = gps_heading

        print(" ")
        # print("waypoints[0] = " + str(waypoints[0]))
        # print("waypoints[10] = " + str(waypoints[10]))
        # print("num of waypoints = " + str(waypoints.shape[0]))
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("num of sampled points = " + str(sampled_waypoints.shape[0]))
        # print("First 10 Sampled Waypoints:")
        # print(sampled_waypoints[:10])

        # Output to file
        # output_directory = ""  # Change this to your desired directory
        # output_filename = os.path.join(output_directory, "sampled_waypoints.csv")

        np.savetxt("sampled_waypoints.csv", sampled_waypoints, delimiter=",", header="X,Y,Heading", comments="")    

        return sampled_waypoints

    def gps_callback(self, msg):
        self.gps = (msg.latitude, msg.longitude, msg.azimuth)
        print(str(self.gps))
        
        


if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass