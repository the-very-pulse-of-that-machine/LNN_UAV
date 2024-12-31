import rospy, math, time
from collections import deque
import numpy as np
import pandas as pd
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest,SetMavFrame
from proxy import posePub, init_sp, cmdPub, velPub, yaw, pid, roll_pitch_yaw, X_Y_H, timeRead, reset_gazebo_simulation, reVel
import detect, inferencer
import communication
import threading,os


def mission():
    init_sp()
    data = np.array([])    
    uavcom = communication.Communication('standard_vtol', '0')
    threading.Thread(target=uavcom.start, args=(), daemon=True).start()
    rospy.sleep(1)   
    

    flag = 0   
    processor = detect.ImageProcessor()   
    lnn_infer = inferencer.Inferencer("1.ckpt")

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

     
    while not rospy.is_shutdown():
        r, p, y = roll_pitch_yaw()
        a, b, height = X_Y_H()
        vx, vy, vz = reVel()
        dya = yaw()
        cx, cy = processor.green_center
        print(f"flag:{flag}  cx{cx}  cy{cy} a{a}, b{b} height{height}")

        rate.sleep()
            
         
        if flag == 0:
            pi = posePub(0, 0, 10, 1, 1)   
            if pi:
                flag = 1
            rate.sleep()

        if flag == 1:
            input = pd.DataFrame([a, b, height, r, p, y, dya, cx, cy])
            control_command = lnn_infer.infer(input)
            
            if abs(cx - 0.5) < 0.12 and abs(cy - 0.5) < 0.12 and abs(p) < 0.1:
                velPub(0, 0, 0, 0)
                flag = -1   
             
            else:
                 
                velPub(control_command.iloc[0, 0], control_command.iloc[0, 1], control_command.iloc[0, 2], control_command.iloc[0, 3])
         
        if flag == -1:
            cmdPub('AUTO.LAND')
            if height < 2.2:
                return 0
 
def main():
    mission()   

if __name__ == "__main__":
    main()

