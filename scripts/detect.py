import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/standard_vtol_0/camera/image_raw', Image, self.image_callback)
        #image ros topic
        self.green_center = None   
        self.running = True   

    def image_callback(self, msg):
        try:
             
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
         
        green_mask = self.detect_green_pixels(cv_image)
        green_pixel_count = np.sum(green_mask > 0)
        
        if green_pixel_count > 500:
            self.green_center = self.calculate_centroid(green_mask, cv_image.shape[1], cv_image.shape[0])
             
            self.mark_center_on_image(cv_image, green_mask, self.green_center)
        else:
            self.green_center = (-1, -1)   
            self.mark_center_on_image(cv_image, green_mask, self.green_center)

    def detect_green_pixels(self, image):
         
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 80, 31])
        upper_green = np.array([77, 255, 185])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask

    def calculate_centroid(self, mask, width, height):
         
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
             
            normalized_cx = cx / width
            normalized_cy = cy / height
            return (normalized_cx, normalized_cy)
        else:
            return (-1, -1)

    def mark_center_on_image(self, image, mask, center):
         
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        green_area = cv2.bitwise_and(image, mask_rgb)
        marked_image = cv2.addWeighted(image, 0.7, green_area, 0.3, 0)

         
        height, width, _ = image.shape

         
        if center != (-1, -1):
             
            pixel_cx = int(center[0] * width)
            pixel_cy = int(center[1] * height)
            cv2.circle(marked_image, (pixel_cx, pixel_cy), 5, (0, 0, 255), -1)

         
        resized_image = cv2.resize(marked_image, (640, 480))   
        cv2.imshow("Processed Image", resized_image)
        cv2.waitKey(1)   

    def shutdown(self):
        self.running = False
        cv2.destroyAllWindows()
