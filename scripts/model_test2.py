import rospy
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, SetMavFrame
from proxy import posePub, init_sp, cmdPub, velPub, yaw, pid, roll_pitch_yaw, X_Y_H, timeRead, reset_gazebo_simulation, reVel
import detect, inferencer
import communication
import threading
import random
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
    lnn_infer = inferencer.Inferencer("ddf3.pth")

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
    an = random.uniform(-1 * np.pi, np.pi)

     
    traj_flag_1 = []   
    traj_flag_3 = []   

     
    while not rospy.is_shutdown():
        r, p, y = roll_pitch_yaw()
        a, b, height = X_Y_H()
        vx, vy, vz = reVel()
        dya = yaw()
        cx, cy = processor.green_center
        print(f"flag:{flag}  cx{cx}  cy{cy} a{a}, b{b} height{height}")

         
        if flag == 1 or flag == -1:
            traj_flag_1.append([a, b, height])
        elif flag == 3:
            traj_flag_1.append([a, b, height])

        rate.sleep()

         
        if flag == 0:
            pi = posePub(35 - 20 * np.cos(an), 40 - 20 * np.sin(an), 10, 1, 1)   
            if pi:
                flag = 1
            rate.sleep()

        if flag == 1:
            if cy > 0.1:   
                velPub(0, 0, 0, 0)
                target = covert([cx * 1280, cy * 720], height, K)
                flag = 3   

            if abs(yaw()) < 0.1:   
                velPub(0, 4, 0, 0)
            else:
                 
                if yaw() > 0:
                    velPub(0, 2, 0, 0.4)
                else:
                    velPub(0, 2, 0, -0.4)

        if flag == 3:
            target = covert([cx * 1280, cy * 720], height, K)
            input = pd.DataFrame([target[1], target[0]])
            control_command = lnn_infer.infer(input)
            print(control_command)
            print(input)

            if height < 2.5 or abs(target[1] ** 2 + target[0] ** 2 < 0.09):
                velPub(0, 0, 0, 0)
                flag = -1   
            else:
                if cx == -1:
                    flag = 1
                else:
                    velPub(1.1 * control_command[0][0], 1.1 * control_command[0][1], control_command[0][2], 0)

         
        if flag == -1:
            velPub(0,0,-1,0)
             
            if height < 2.2:
                print(f'd:{((a - 35) ** 2 + (b - 40) ** 2) ** 0.5}')
                break

     
    plot_trajectory(traj_flag_1)

 
def plot_trajectory(traj_flag_1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

     
    traj_flag_1 = np.array(traj_flag_1)

     
    ax.plot(traj_flag_1[:, 0], traj_flag_1[:, 1], traj_flag_1[:, 2], label="", color='b')
    ax.scatter(35, 40, 2, label="target", color="r")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')
    ax.set_title('LNN Trajectory')

    ax.legend()
    plt.show()

 
def main():
    mission()   

if __name__ == "__main__":
    main()

