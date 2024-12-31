import rospy, math, time
from collections import deque
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest,SetMavFrame
from proxy import posePub, init_sp, cmdPub, velPub, yaw, pid, roll_pitch_yaw, X_Y_H, timeRead, reset_gazebo_simulation, reVel
import detect
import communication
import threading,os,random
from target_calc import *
import matplotlib.pyplot as plt

K = np.array([[369.5021, 0, 640],
              [0, 369.5021, 360],
              [0, 0, 1]]) 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'data_dr')

if not os.path.exists(save_dir):  
    os.makedirs(save_dir)

npy_file = os.path.join(save_dir, 'neo.npy')

def mission():
    init_sp()
    if os.path.exists(npy_file):
        data = np.load(npy_file)
    else:
        data = np.empty((0, 9))
    
    uavcom = communication.Communication('standard_vtol', '0')
    threading.Thread(target=uavcom.start, args=(), daemon=True).start()
    rospy.sleep(1) 

    flag = 0  
    processor = detect.ImageProcessor()  

    print("init done")

     
    errx, inx = 0, 0
    erry, iny = 0, 0
    KP4, KI4, KD4 = 0.12, 0.005,0 
    KP5, KI5, KD5 = 0.12, 0.005,0 
    
    traj_flag_1 = []   
    traj_flag_3 = []   
    
    vx = 0
    vy = 0
    rate = rospy.Rate(20)   
    an = random.uniform(-1*np.pi, np.pi)
     
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
        
        if flag == 1 or flag == -1:
            traj_flag_1.append([a, b, height])
        elif flag == 3:
            traj_flag_1.append([a, b, height])
        
        if processor.green_center is not None:
             
            cx, cy = processor.green_center
            print(f"flag:{flag}  cx{cx}  cy{cy} a{a}, b{b} height{height}")

        rate.sleep()
            
         
        if flag == 0:
            pi = posePub(35 - 20 * np.cos(an) , 40 - 20 * np.sin(an), 10, 1, 1)   
            if pi:
                flag = 1
            rate.sleep()

         
        if flag == 1:
            if cy > 0.1:   
                velPub(0, 0, 0, 0)
                 
                flag = 3   
            
            if abs(yaw()) < 0.1:   
                velPub(0, 4, 0, 0)
            else:
                 
                if yaw() > 0:
                    velPub(0, 2, 0, 0.5)
                    wa = 0.2
                else:
                    velPub(0, 2, 0, -0.5)
                    wa = -0.2
         
        if flag == 3:
            if cx == -1:   
                flag = 1
            else:
                target = covert([cx*1280, cy*720],height,K)
            
            da, db = target[0] - a, target[1] - b
            da1 = da*np.cos(y)+db*np.sin(y)
            db1 = da*np.sin(-1*y)+db*np.cos(y)
            
             
            outp_vel, errx, inx = pid(target[1], 0, KP4, KI4, KD4, errx, inx)
            outy_vel, erry, iny = pid(target[0], 0, KP5, KI5, KD5, erry, iny)

             
            if height < 3:  
                velPub(0, 0, 0, 0)
                flag = -1   
                outp_vel = 0
                outy_vel = 0
                
             
            else:
                 
                velPub(outy_vel, outp_vel, -0.3,  0)
                rospy.loginfo(f"p {outp_vel}, yaw {outy_vel}")

         
        if flag == -1:
             
            velPub(0,0,-0.2,0)
            if height < 2.2:
                user_input = input("Do you want to save the data? (y/n): ").strip().lower()
                plot_trajectory(traj_flag_1)
                if user_input == 'y':   
                    np.save(npy_file, data)
                    print("Data saved.")
                    return 0
                elif user_input == 'n':   
                    print("Data not saved.")
                    return 0
                else:   
                    print("Invalid input. Please enter 'y' or 'n'.")
                     
                    return 0
        
        if flag == 3 or flag == -1:
         
            vx, vy, vz = reVel()
            dya = yaw()
            data = np.append(data, [outy_vel, outp_vel, vz, target[1], target[0]])
        

    

 
def plot_trajectory(traj_flag_1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

     
    traj_flag_1 = np.array(traj_flag_1)

     
    ax.plot(traj_flag_1[:, 0], traj_flag_1[:, 1], traj_flag_1[:, 2], label="", color='b')
    ax.scatter(35, 40, 2, label="target", color="r")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')
    ax.set_title('PID Trajectory')

    ax.legend()
    plt.show()

 
def main():
    mission()   

if __name__ == "__main__":
    main()

