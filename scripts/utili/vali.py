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

     
    uavcom = communication.Communication('standard_vtol', '0')
    threading.Thread(target=uavcom.start, args=(), daemon=True).start()
    rospy.sleep(1)   

     
    targetx = 10
    targety = 10

    flag = 0   
    processor = detect.ImageProcessor()   

    print("init done")

     
    errx, inx = 0, 0
    erry, iny = 0, 0
    KP4, KI4, KD4 = 3, 0., 0   
    KP5, KI5, KD5 = 3, 0., 0   
    
    vx = 0
    vy = 0
    rate = rospy.Rate(20)   

     
    cmdPub('OFFBOARD')
    print("taking off")
    time.sleep(3)   
    for i in range(0, 5):
        cmdPub('ARM')   
        cmdPub('OFFBOARD')
        cmdPub('multirotor')   

    cunt = 0   

    while not rospy.is_shutdown():
         
        r, p, y = roll_pitch_yaw()
         
        a, b, height = X_Y_H()
        if processor.green_center is not None:
             
            cx, cy = processor.green_center
            print(f"flag:{flag}  cx{cx}  cy{cy} height{height}")

        rate.sleep()
         
        if flag == 0:
            p = posePub(0, 0, 10, 1, 1)   
            if p:
                flag = 1
            rate.sleep()
         
        if flag == 1:
            if abs(yaw()) < 0.1:   
                flag = 2
            else:
                 
                if yaw() > 0:
                    velPub(0, 0, 0, 0.5)
                else:
                    velPub(0, 0, 0, -0.5)
         
        if flag == 2:
            velPub(0, 4, 0, 0)   
            if cx != -1:   
                
                target = CoordinateTransfer(RMatrixCalc(r, p, 0), cx*1280, cy*720, a, b, height, K)
                velPub(0, 0, 0, 0)
                flag = 2   
                errx = 0
                inx = 0 
                erry = 0
                iny = 0
                
                vx = 0
                vy = 0
         
        if flag == -1:
            p = posePub(0, 0, 0, 1, 0.6)   
            cmdPub('AUTO.LAND')   
            if height < 2.2:   
                processor.shutdown()   
                return 1
 
def main():
    while not rospy.is_shutdown():
        mission()   

if __name__ == "__main__":
    main()

