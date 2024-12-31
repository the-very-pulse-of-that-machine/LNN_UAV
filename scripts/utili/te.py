import rospy, math, time
from collections import deque
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from proxy import posePub, init_sp, cmdPub, velPub, yaw, pid, roll_pitch_yaw, X_Y_H, timeRead, reset_gazebo_simulation
import detect
import communication
import threading
from target_calc import *

K = np.array([[369.5021, 0, 640],
              [0, 369.5021, 360],
              [0, 0, 1]]) 

 
def Vert2Pol(inx, iny):
     
    if iny == 0:
        iny = 0.0001
        
     
    outp = math.sqrt(inx**2 + iny**2) * abs(iny) / iny
    
     
    ang = math.atan2(inx, abs(iny))
    if outp > 5:   
        outp = 4 
    return outp, -1 * ang

 
def pn(x):
    if x > 0:
        return 1
    else:
        return -1

 
def mission():
     
    init_sp()
    rate = rospy.Rate(20)
     
    uavcom = communication.Communication('standard_vtol', '0')
    threading.Thread(target=uavcom.start, args=(), daemon=True).start()
    rospy.sleep(1)   

    processor = detect.ImageProcessor() 
    print("init done")


     
    while not rospy.is_shutdown():
         
        velPub(0,0,0,0.8)
        r, p, y = roll_pitch_yaw()
        print(X_Y_H())
        print(roll_pitch_yaw())
         
        a, b, height = X_Y_H()
        if processor.green_center is not None:
             
            cx, cy = processor.green_center
            print(f'cxcy ; {cx}, {cy} ')
            rate.sleep()
            if cx != -1:   
                
                target = covert([cx*1280, cy*720],height,K)
                tarture = CoordinateTransfer(RMatrixCalc(r, p, y), cx*1280, cy*720, a, b, height, K)
                print(f'target: {tarture}')
                print(f'tart{target}')
                print(f'true: {a-target[1]} {b-target[0]}')



 
def main():
    while not rospy.is_shutdown():
        mission()   

if __name__ == "__main__":
    main()

