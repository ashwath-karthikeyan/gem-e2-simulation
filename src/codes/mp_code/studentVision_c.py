import time
import math
import numpy as np
import cv2
# import pandas as pd
import rospy
# import matplotlib.pyplot as plt

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

# global detection_counter

# detection_counter = 0

class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE 
        # Uncomment this line for lane detection of GEM car in Gazebo
        self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for the gem car rosbag
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #cv2.imshow("cvimage",cv_image)
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)
            

    def gradient_thresh(self, img, thresh_min, thresh_max):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        # imgplot = plt.imshow(img)
        #plt.show()
        #1. Convert the image to gray scale
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #2. Gaussian blur the image
        blur = cv2.GaussianBlur(gray,(3,3),0)
        #cv2.imshow("img is ", blur)
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1,0,ksize =3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0,1,ksize =3)
        #added = 0.5*sobelx + 0.5*sobely
        added = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        #cv2.imshow("added is", added)
        # pd.DataFrame(added).to_csv('sample.csv') 
        scaled_sobel = np.zeros_like(added)
        scaled_sobel[(added >=thresh_min) & (added<= thresh_max)] = 1
        #thresh = cv2.threshold(added, 10,100, cv2.THRESH_BINARY)
        #cv2.imshow("scaled sobel is", scaled_sobel*255)
        #cv2.imshow("thresh is", thresh)
        #cv2.waitKey(0)
        return scaled_sobel

    def color_thresh(self, img, thresh):
        
        """
        Convert RGB to HSL and threshold to binary image using S channel
        
        #1. Convert the image from RGB to HSL
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        #L = hls[:,:,2]
        #h = hls[:,:,0]
        s = hls[:,:,1]

        cv2.imshow("s is ", s)
        #cv2.imshow("h is ", h)
        #cv2.imshow("l is ", v)

        #2. Apply threshold on S channel to get binary image
        col_s = np.zeros_like(s)
        #col_s = np.zeros_like(s)

        #h_t = np.zeros_like(h)
        #h_t[(h >=100)] = 1
        col_s[(s >= 200) & (s <= 255)] = 1
        #col_s[(s <=15)] = 1

        #thresh = cv2.inRange(s,50,150)
        #cv2.imshow("v thresh is ", 255*col_v)
        cv2.imshow("s thresh is ", 255*col_s)
        #cv2.waitKey(0)
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        l = lab[:,:,2]
        s = lab[:,:,0]
        a = lab[:,:,1]
        #cv2.imshow("s is ", s)
        #cv2.imshow("l is ", l)
        #cv2.imshow("a is ", a)
        col_s = np.ones_like(s)
        col_s[(s >= 145) & (s <= 255)] = 0
        #cv2.imshow("s thresh is ", 255*col_s)
        #cv2.waitKey()
        return col_s

    def combinedBinaryImage(self, img):
        # Get combined binary image from color filter and sobel filter
        # """
        #1. Apply sobel filter and color filter on input image
        SobelOutput = self.gradient_thresh(img, thresh_min= 25, thresh_max=175)
        ColorOutput = self.color_thresh(img, thresh=(80, 150))
        #2. Combine the outputs
        ## Here you can use as many methods as you want. 

        ## TODO
        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        # Remove noise from binary image
        #binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        #cv2.imshow("combined is ", 255*ColorOutput)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        return binaryImage




    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image  
        """
        #1. Visually determine 4 source points and 4 destination points
        #rosbag1 points
        #img = np.uint8(img)
        dim= np.shape(img)
        #imgplot = plt.imshow(img)
        #plt.show()
        # print("dim is ", dim)
        #cv2.imshow("img",img)
        #pt_A = [400, 600]
        #pt_B = [958,600]
        #pt_C = [728, 220]
        #pt_D = [300, 220]
        # tl = [0, 360]
        # bl = [0, 720]
        # tr = [1280, 360]
        # br = [1280, 720]
        tl = [0, 400]
        bl = [0, 720]
        tr = [1280, 400]
        br = [1280, 720]
        print(dim[0],dim[1])
        input_pts = np.float32([tl, bl, tr, br])
        dim= np.shape(img)
        pt_A = [0,0]
        pt_B = [0, 720]
        pt_C = [1280, 0]
        pt_D = [1280, 720]
        output_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        #3. Generate warped image in bird view using cv2.warpPerspective()
        warped_img = cv2.warpPerspective(img,M,(1280,720),flags=cv2.INTER_LINEAR)
        # warped_img = cv2.warpPerspective(img,M,(640,480),flags=cv2.INTER_LINEAR)
        #
        # cv2.imshow("warp",warped_img)
        # cv2.waitKey(0)
        Minv = cv2.getPerspectiveTransform(output_pts,input_pts)
        
        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init arg
    
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()


    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
