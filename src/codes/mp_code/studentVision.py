import time
import math
import numpy as np
import cv2
import rospy
import torch


from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from demo import detect






class lanenet_detector():
   def __init__(self):


       self.bridge = CvBridge()
       # NOTE
       # Uncomment this line for lane detection of GEM car in Gazebo
    #    self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
       # Uncomment this line for lane detection of videos in rosbag
    #    self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
       self.sub_image = rospy.Subscriber('zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
       self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
    #    self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
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
       mask_image= self.detection(raw_img)


       if mask_image is not None:
           # Convert an OpenCV image into a ROS image message
           out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
        #    out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')


           # Publish image message in ROS
           self.pub_image.publish(out_img_msg)
        #    self.pub_bird.publish(out_bird_msg)




   def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
       """
       Apply sobel edge detection on input image in x, y direction
       """
       #1. Convert the image to gray scale
       #2. Gaussian blur the image
       #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
       #4. Use cv2.addWeighted() to combine the results
       #5. Convert each pixel to uint8, then apply threshold to get binary image


       ## TODO
       ####
       frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


       blur = cv2.GaussianBlur(frame_gray,(15, 15),0)


       x_der = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
       y_der = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)


       # x_grad = cv2.convertScaleAbs(x_der)
       # y_grad = cv2.convertScaleAbs(y_der)
          
       grad_temp = cv2.addWeighted(x_der, 0.8, y_der, 0.2, 0.0)


       grad = cv2.convertScaleAbs(grad_temp) # converts back to CV_8U




       ret, binary_img = cv2.threshold(grad, 25, 255, cv2.THRESH_TOZERO)
       ret, binary_img2 = cv2.threshold(binary_img, 75, 255, cv2.THRESH_TOZERO_INV)
       ret, binary_img3 = cv2.threshold(binary_img2, 0, 255, cv2.THRESH_BINARY)
      
       # ret, binary_img = cv2.threshold(grad, 25, 255, cv2.THRESH_TOZERO)
       # ret, binary_img2 = cv2.threshold(binary_img, 100, 255, cv2.THRESH_TOZERO_INV)
       # ret, binary_img2 = cv2.threshold(grad, 100, 255, cv2.THRESH_BINARY)
       # ret, binary_img3 = cv2.threshold(binary_img2, 25, 255, cv2.THRESH_BINARY)
      


       binary_output = binary_img3
      
       # cv2.imshow("frame3", binary_output)
       # cv2.waitKey(0)
       # cv2.destroyAllWindows()
       # # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
       # blur = cv2.GaussianBlur(gray,(5,5),0)


       # x_der = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
       # y_der = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)


       # gradmag = np.sqrt(x_der**2 + y_der**2)
       # scale_factor = np.max(gradmag)/255
       # gradmag = (gradmag/scale_factor).astype(np.uint8)
       # binary_output = np.zeros_like(gradmag)
       # binary_output[(gradmag >= 25) & (gradmag <= 100)] = 255


       return binary_output




   def color_thresh(self, img, thresh=(100, 255)):
       """
       Convert RGB to HSL and threshold to binary image using S channel
       """
       #1. Convert the image from RGB to HSL
       #2. Apply threshold on S channel tso get binary image
       #Hint: threshold on H to remove green grass
       ## TODO


       thresh=(100, 255)
       green_thresh = (0,15)
       sky_thresh = (36, 179)
       hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)




       mask = cv2.inRange(hls_image, (green_thresh[0],0,0), (green_thresh[1], 155,thresh[1])) #
       mask_image = cv2.bitwise_not (mask, mask)
       last_img = cv2.bitwise_and (img, img, mask=mask_image)


       #Filter Sky255,
       mask = cv2.inRange(hls_image, (sky_thresh[0],0,0), (sky_thresh[1], 155,thresh[1]))
       mask_image = cv2.bitwise_not (mask, mask)
       last_img2 = cv2.bitwise_and (last_img, last_img, mask=mask_image)


       #Get low clarity ones
       yellow_mask = cv2.inRange(hls_image, (15,100,15), (25, 125,85))
       # last_img3 = cv2.bitwise_and (last_img2, last_img2, mask=mask)


       # #Get low clarity white lanes
       white_mask = cv2.inRange(hls_image, (0,100,2), (255, 170,5))
       new_mask = cv2.bitwise_or(yellow_mask, white_mask)
       binary_img = cv2.bitwise_and(last_img2, last_img2, mask=new_mask)


       ret, binary_img = cv2.threshold(binary_img[:,:,2], 0, 255, cv2.THRESH_BINARY)




       ##
       # cv2.imshow("color", binary_img)
       # cv2.waitKey(0)
       # cv2.destroyAllWindows()


       return binary_img




   def combinedBinaryImage(self, img):
       """
       Get combined binary image from color filter and sobel filter
       """
       #1. Apply sobel filter and color filter on input image
       #2. Combine the outputs
       ## Here you can use as many methods as you want.


       ## TODO


       ####


       SobelOutput = self.gradient_thresh(img, 25, 100)
       ColorOutput = self.color_thresh(img, thresh=(100, 255))




       binaryImage = np.zeros_like(SobelOutput)
       binaryImage[(ColorOutput==255)|(SobelOutput==255)] = 255


       # binaryImage = np.zeros_like(SobelOutput)
       # binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
       # Remove noise from binary image
       binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=5,connectivity=20)
       binaryImage=np.uint8(binaryImage)*255




       return binaryImage




   def perspective_transform(self, img, verbose=False):
       """
       Get bird's eye view from input image
       """
       #1. Visually determine 4 source points and 4 destination points
       #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
       #3. Generate warped image in bird view using cv2.warpPerspective()


    #    pts1 = np.float32([ [250, 290], [400, 290], [35, 388], [660, 388]])
    #    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])


    #    pts1 = np.float32([ [450, 250], [711, 250], [300, 344], [800, 344]]) # for 0011
    #    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

       pts1 = np.float32([ [450, 250], [711, 250], [250, 344], [890, 344]]) #for 0056
       pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])



    #    pts1 = np.float32([ [580, 353], [670, 353], [176, 656], [940, 656]]) #for 0830 1
    #    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])


    #    pts1 = np.float32([ [540, 400], [750, 400], [323, 624], [934, 624]]) #for 0830
    #    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])


    #    pts1 = np.float32([ [582, 347], [660, 347], [136, 704], [960, 704]]) #for 0830 2
    #    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])


      
       matrix = cv2.getPerspectiveTransform(pts1, pts2)
       result = cv2.warpPerspective(img, matrix, (640, 480))


       M = matrix
       warped_img = result
       Minv = np.linalg.inv(matrix)
       # Minv = np.linalg.inv(matrix)


       ## TODO


       ####


       return warped_img, M, Minv




   def detection(self, img):


    #    binary_img = self.combinedBinaryImage(img)
    #    img_birdeye, M, Minv = self.perspective_transform(binary_img)

       cv2.imwrite('camera_image.jpg', img)
       with torch.no_grad():
          new_img = detect('camera_image.jpg')



    #    # cv2.imshow("frame3", binary_img)
    #    # cv2.waitKey(0)
    #    # cv2.destroyAllWindows()
      
    #    # cv2.imshow("frame3", img_birdeye)
    #    # cv2.waitKey(0)
    #    # cv2.destroyAllWindows()cv2.imshow("color", binary_img)
    #    # cv2.waitKey(0)
    #    # cv2.destroyAllWindows()




    #    if not self.hist:
    #        # Fit lane without previous result
    #        ret = line_fit(img_birdeye)
    #        left_fit = ret['left_fit']
    #        right_fit = ret['right_fit']
    #        nonzerox = ret['nonzerox']
    #        nonzeroy = ret['nonzeroy']
    #        left_lane_inds = ret['left_lane_inds']
    #        right_lane_inds = ret['right_lane_inds']


    #    else:
    #        # Fit lane with previous result
    #        if not self.detected:
    #            ret = line_fit(img_birdeye)


    #            if ret is not None:
    #                left_fit = ret['left_fit']
    #                right_fit = ret['right_fit']
    #                nonzerox = ret['nonzerox']
    #                nonzeroy = ret['nonzeroy']
    #                left_lane_inds = ret['left_lane_inds']
    #                right_lane_inds = ret['right_lane_inds']


    #                left_fit = self.left_line.add_fit(left_fit)
    #                right_fit = self.right_line.add_fit(right_fit)


    #                self.detected = True


    #        else:
    #            left_fit = self.left_line.get_fit()
    #            right_fit = self.right_line.get_fit()
    #            ret = tune_fit(img_birdeye, left_fit, right_fit)


    #            if ret is not None:
    #                left_fit = ret['left_fit']
    #                right_fit = ret['right_fit']
    #                nonzerox = ret['nonzerox']
    #                nonzeroy = ret['nonzeroy']
    #                left_lane_inds = ret['left_lane_inds']
    #                right_lane_inds = ret['right_lane_inds']


    #                left_fit = self.left_line.add_fit(left_fit)
    #                right_fit = self.right_line.add_fit(right_fit)


    #            else:
    #                self.detected = False


    #        # Annotate original image
    #        bird_fit_img = None
    #        combine_fit_img = None
    #        if ret is not None:
    #            bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
    #            combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
    #        else:
    #            print("Unable to detect lanes")


       return new_img




if __name__ == '__main__':
   # init args
   rospy.init_node('lanenet_node', anonymous=True)
   lanenet_detector()
   while not rospy.core.is_shutdown():
       rospy.rostime.wallsleep(0.5)





