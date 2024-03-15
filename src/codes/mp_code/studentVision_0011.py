import time
import math
import numpy as np
import cv2
import rospy
import matplotlib.pyplot as plt

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for the gem car rosbag
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        #cv2.imshow('graycsale image',self.sub_image)
        #print(self.sub_image)
        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        #cv2.waitKey(0)
        
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


    def gradient_thresh(self, img, thresh_min = 25, thresh_max = 100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image

        ## TODO
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_blur = cv2.GaussianBlur(img_gray,(11,11),0)
        # cv2.imshow('org', img_gray_blur)

        grad_x = cv2.Sobel(img_gray_blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray_blur, cv2.CV_16S, 0, 1, ksize=3)
 

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
 
        # cv2.imshow('grad_x', abs_grad_x)

        grad = cv2.addWeighted(abs_grad_x, 0.9, abs_grad_y, 0.1, 0)
        _,binary_output1 = cv2.threshold(grad,50,255,cv2.THRESH_BINARY)
        _,binary_output2 = cv2.threshold(grad,150,255,cv2.THRESH_BINARY)
        binary_output = binary_output1 - binary_output2
        #cv2.imshow('low thres', binary_output1)
        #cv2.imshow('high thres', binary_output2)
        # cv2.imshow('thres', binary_output)

        # cv2.waitKey()

        #cv2.imshow('img_gray_blur', img_gray_blur)

        #cv2.imshow('grad', grad)
        
        #cv2.imshow('grad', binary_output)
        #cv2.waitKey()

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO
        hsl_img=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsl_img_h=hsl_img[:,:,0]
        hsl_img_l=hsl_img[:,:,1]
        hsl_img_s=hsl_img[:,:,2]
        _,binary_output = cv2.threshold(hsl_img_s,200,255,cv2.THRESH_TOZERO_INV)
        
        #180
        # cv2.imshow('hsl_img_h', hsl_img_h)

        #cv2.imshow('hsl_img_l', hsl_img_l)
        
        #cv2.imshow('hsl_img_s', hsl_img_s)

        # cv2.imshow('color', binary_output)
        #cv2.imshow('color', np.float32(color_t_img))
        #cv2.imshow('gradient', np.float32(gradient_t_img))


        # cv2.waitKey()

        ####

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO

        ####
        '''
        h = img.shape[0]
        w = img.shape[1]
        polygons=np.array([[(w,0),(0,0),(0,h//2),(w,h//2)]])
        mask=np.zeros_like(img)
        cv2.fillPoly(mask,np.int32([polygons]),255)
        masked_img=cv2.bitwise_and(img,mask)
        cv2.imshow('masked_img', np.float32(masked_img))
        #print(img_crop.shape)
        #print(warped_img.shape)
        cv2.waitKey()
        

        color_t_img=self.color_thresh(masked_img)
        gradient_t_img=self.gradient_thresh(masked_img)
        '''
        
        color_t_img=self.color_thresh(img)
        gradient_t_img=self.gradient_thresh(img)
        #h_0 = img.shape[0]
        

        binaryImage = np.zeros_like(color_t_img)
        binaryImage[(gradient_t_img==255)|(color_t_img==255)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        #h_0 = binaryImage.shape[0]
        #binaryImage=binaryImage[h_0//2:h_0, :]
        
        #cv2.imshow('merge', binaryImage)
        #cv2.imshow('color', np.float32(color_t_img))
        #cv2.imshow('gradient', np.float32(gradient_t_img))


        #cv2.waitKey()
        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        h = img.shape[0]
        w = img.shape[1]

        # tl=(500, 210)
        # bl=(113, 350)
        # tr=(750, 210)
        # br=(1000, 350)

        tl=(440, 250)
        bl=(250, 350)
        tr=(730, 270)
        br=(810, 350)



        pts1=np.array([tl,tr,br,bl],np.float32)
        pts2=np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
        M=cv2.getPerspectiveTransform(pts1,pts2)
        Minv=cv2.getPerspectiveTransform(pts2,pts1)
        warped_img=cv2.warpPerspective(np.float32(img),M,(w, h))
        #cv2.circle(np.float32(img_crop),tl,100,(255,0,255),-1)
        #cv2.circle(np.float32(img_crop),tr,100,(255,0,255),-1)
        #cv2.circle(np.float32(img_crop),br,100,(255,0,255),-1)
        #cv2.circle(np.float32(img_crop),bl,100,(255,0,255),-1)
        # cv2.imshow('test', np.float32(img_crop))
        # cv2.imshow('org', np.float32(img))
        # cv2.imshow('warped_img', np.float32(warped_img))
        #print(img_crop.shape)
        #print(warped_img.shape)
        # cv2.waitKey()
        ####
        #print('pts  ',pts1, pts2)
        # plt.subplot(121), plt.imshow(img), plt.title('Input img')
        # plt.show()
        return warped_img, M, Minv


    def detection(self, img):
        cv2.imwrite('warped.jpg', img)

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)
        # cv2.imshow('bimary', np.float32(binary_img))
        # cv2.imshow('bird', np.float32(img_birdeye))
        # cv2.waitKey()
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
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
