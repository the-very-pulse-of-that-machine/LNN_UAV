 

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Pose, TwistStamped
from pyquaternion import Quaternion
from gazebo_msgs.msg import ModelStates

target_x = 0
target_y = 0

def States_cb(msg):
    global target_x, target_y
    try:
        index = msg.name.index('target_true')
    except ValueError:
        rospy.logerr("target_true not found in model states")
        return
    position = msg.pose[index].position
     
    target_x = position.x-5
    target_y = position.y

def q2yaw(q):
    if isinstance(q, Quaternion):
        rotate_z_rad = q.yaw_pitch_roll[0]
    else:
        q_ = Quaternion(q.w, q.x, q.y, q.z)
        rotate_z_rad = q_.yaw_pitch_roll[0]
    return rotate_z_rad


def yaw(target_pose = [-95, 0], vtol_pose = [1, 0]):
    modulo_length = np.linalg.norm(np.array(target_pose)) * np.linalg.norm(np.array(vtol_pose))
    angel_direction = np.cross(np.array(target_pose), np.array(vtol_pose))
    a = np.random.normal(0, 1, 1)
    while(a[0]<-0.1 or a[0]>0.1):
        a = np.random.normal(0, 1, 1)
    angel = np.arccos(np.dot(np.array(target_pose), np.array(vtol_pose)) / modulo_length) + a[0]
    if (angel_direction > 0):
        angel = - angel
    if(angel > np.pi):
        angel -= 2 * np.pi
    if(angel < -np.pi):
        angel += 2 * np.pi

    pub_pose = Pose()
    pub_pose.position.x, pub_pose.position.y, pub_pose.position.z = angel, 0.0 , 0.0
    pub_pose.orientation.x, pub_pose.orientation.y, pub_pose.orientation.z, pub_pose.orientation.w = 0.0, 0.0, 0.0, 0.0
    zhihang.publish(pub_pose)

def zhihang_vtol_position_cb(msg):
    global target_x, target_y
    center_x = -1200
    center_y = -1200
    radius = 1000

    vtol_standard_yaw = q2yaw(msg.orientation)
    current_pose = msg
    vtol_x = current_pose.position.x
    vtol_y = current_pose.position.y
    distance_to_center = np.sqrt((vtol_x - center_x)**2 + (vtol_y - center_y)**2)
    vtol_pose = [np.cos(vtol_standard_yaw), np.sin(vtol_standard_yaw)]
    yaw(target_pose=[target_x - 5 - vtol_x, target_y - vtol_y], vtol_pose=vtol_pose)

if __name__ == "__main__":
    rospy.init_node('zhihang2')
    rospy.Subscriber("/gazebo/model_states", ModelStates, States_cb, queue_size=1)
    rospy.Subscriber("/gazebo/zhihang/standard_vtol/local_position", Pose, zhihang_vtol_position_cb, queue_size=1)
    zhihang = rospy.Publisher('/zhihang/standard_vtol/angel', Pose, queue_size=1)
    rospy.spin()

