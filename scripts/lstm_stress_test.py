import rospy, math, time
from collections import deque
import numpy as np
import pandas as pd
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest,SetMavFrame
from proxy import posePub, init_sp, cmdPub, velPub, yaw, pid, roll_pitch_yaw, X_Y_H, timeRead, reset_gazebo_simulation, reVel
import detect, lstminferencer
import communication
import threading,os,random
from target_calc import *

K = np.array([[369.5021, 0, 640],
              [0, 369.5021, 360],
              [0, 0, 1]])

def mission():
    init_sp()
    data = np.array([])
    
     
    uavcom = communication.Communication('standard_vtol', '0')
    threading.Thread(target=uavcom.start, args=(), daemon=True).start()
    rospy.sleep(1)   
    

    flag = 0   
    processor = detect.ImageProcessor()   
    lstm_infer = lstminferencer.LSTMInferencer("lstm_model.pth")

    print("init done")
    rate = rospy.Rate(20)   

     
    cmdPub('OFFBOARD')
    print("taking off")
    time.sleep(1)   
    for i in range(0, 5):
        cmdPub('ARM')   
        cmdPub('OFFBOARD')
        cmdPub('multirotor')   

    cunt = 0   
    noise_std_dev = 0.1
    an = random.uniform(-1*np.pi, np.pi) #noise
     
    while not rospy.is_shutdown():
        r, p, y = roll_pitch_yaw()
        a, b, height = X_Y_H()
        vx, vy, vz = reVel()
        dya = yaw()
        cx, cy = processor.green_center
        print(f"flag:{flag}  cx{cx}  cy{cy} a{a}, b{b} height{height}")

        rate.sleep()
            
         
        if flag == 0:
            pi = posePub(35 - 20 * np.cos(an) , 40 - 20 * np.sin(an), 10, 1, 1)   
            if pi:
                flag = 1
            rate.sleep()
        
        if flag == 1:
            if cx != -1:   
                velPub(0, 0, 0, 0)
                target = covert([cx*1280, cy*720],height,K)
                
                flag = 3   
            
            if abs(yaw()) < 0.1:   
                velPub(0, 4, 0, 0)
            else:
                 
                if yaw() > 0:
                    velPub(0, 4, 0, 0.4)
                else:
                    velPub(0, 4, 0, -0.4)
        if flag == 3:
            target = covert([cx*1280, cy*720],height,K)
            input = pd.DataFrame([target[1], target[0]])
            target[1] *= 1+np.random.normal(0, noise_std_dev)
            target[0] *= 1+np.random.normal(0, noise_std_dev)
            control_command = lstm_infer.infer(input)
            print(control_command)
            print(input)
            
            if height < 2.5: 
                velPub(0, 0, 0, 0)
                flag = -1   
             
            else:
                if cx == -1:
                    flag = 2
                 
                else:
                    velPub(control_command[0][0], control_command[0][1], control_command[0][2], 0)


         
        if flag == -1:
            cmdPub('AUTO.LAND')
            if height < 2.2:
                print(f'd:{((a-35)**2 + (b-40)**2)**0.5}')
                return 0
        
        
 
def main():
    mission()   

if __name__ == "__main__":
    main()

