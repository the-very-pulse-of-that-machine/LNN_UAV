 
import rospy
import time, math, random
from geometry_msgs.msg import PoseStamped, Pose, Twist, TwistStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import String, Bool, Float64
from std_srvs.srv import Empty

current_state = State()
current_pose = PoseStamped()
current_vel = TwistStamped()
current_truePose = Pose()
current_angel = 0.0
current_thunder = [0.0,0.0]
current_roll, current_pitch ,current_yaw = 0, 0, 0
cx, cy = 0,0
wx, wy = 0,0
current_sim_time = 0.0

def state_cb(msg):
    global current_state
    current_state = msg

def pose_cb(msg):
    global current_pose
    current_pose = msg
    current_pose.position.x = current_pose.position.x 
    current_pose.position.y = current_pose.position.y 
    current_pose.position.z = current_pose.position.z 
    global current_roll, current_pitch, current_yaw
    current_roll, current_pitch ,current_yaw = quaternion_to_euler(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w)
    

def angel_cb(msg):
    global current_angel
    current_angel = msg.position.x

def thunder_cb(msg):
    global current_thunder
    current_thunder[0] = msg.position.x
    current_thunder[1] = msg.position.y

def vel_cb(msg):
    global current_vel
    current_vel = msg

def cxcy(msg):
    global cx, cy, wx, wy
    cx = msg.position.x
    cy = msg.position.y
    wx, wy = msg.orientation.x, msg.orientation.y

def recxcy():
    return cx, cy, wx, wy

def clock_callback(data):
    global current_sim_time
    current_sim_time = data.clock.secs

def timeRead():
    return current_sim_time

def init_sp():
    rospy.init_node("offb_node_py")
    state_sub = rospy.Subscriber("/standard_vtol_0/mavros/state", State, callback=state_cb)
    pose_sub = rospy.Subscriber("/gazebo/zhihang/standard_vtol/local_position", Pose, callback=pose_cb)
    vel_sub = rospy.Subscriber("/standard_vtol_0/mavros/local_position/velocity_local", TwistStamped, callback=vel_cb)
    
    angel_sub = rospy.Subscriber("/zhihang/standard_vtol/angel", Pose, callback=angel_cb)
    thunder_sub = rospy.Subscriber("/zhihang/thunderstorm", Pose, callback=thunder_cb)
        
    cxcy_sub = rospy.Subscriber("/object_center", Pose, callback=cxcy)
    
    time_sub = rospy.Subscriber('/clock', Float64, callback=clock_callback)
        
    global state_pub
    state_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd', String, queue_size=3)
    
    global pose_enu_pub 
    pose_enu_pub = rospy.Publisher('/standard_vtol_0/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    
    global pose_flu_pub 
    pose_flu_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd_pose_flu', Pose, queue_size=1)
    
    global vel_pub
    vel_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd_vel_flu', Twist, queue_size=1)
    
    global arrive_pub
    arrive_pub = rospy.Publisher('/arrive', Bool, queue_size=1)
    
    global ship_pub
    ship_pub = rospy.Publisher('/ship', Bool, queue_size=1)
    
    
    
def cmdPub(state):
 
    cmd = String()
    cmd = state
    state_pub.publish(cmd)
    

def posePub(x, y, z, m, ac): 
    if m == 1:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose_enu_pub.publish(pose)
        if -1*ac < pose.pose.position.x - current_pose.position.x < ac and -1*ac < pose.pose.position.y  - current_pose.position.y <ac and -0.3 < pose.pose.position.z - current_pose.position.z < 0.4 :

            return 1
        else:
            return 0
    if m == 0:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose_flu_pub.publish(pose)
        if -1*ac < pose.pose.position.x - current_pose.pose.position.x < ac and -1*ac < pose.pose.position.y  - current_pose.pose.position.y <ac and -0.2 < pose.pose.position.z - current_pose.pose.position.z < 0.2 :
            return 1
        else:
            return 0
            
        
def velPub(x,y,z,yaw):
    vel = Twist()
    vel.linear.x = x
    vel.linear.y = -1*y
    vel.linear.z = z
    vel.angular.x = 0
    vel.angular.y = 0
    vel.angular.z = yaw
    vel_pub.publish(vel)
    
def missionPub(x):
    if x == 1:
        is_arrived = True
        arrive_pub.publish(is_arrived)
    if x == 2:
        is_ship = True
        ship_pub.publish(is_ship)
        
def globalVel():
    return current_vel.twist.linear.x, current_vel.twist.linear.y

def yaw():
    return current_angel

def roll_pitch_yaw():
    return current_roll, current_pitch, current_yaw
    
def X_Y_H():
    return current_pose.position.x, current_pose.position.y, current_pose.position.z
    
def pid(current_value, setpoint, Kp, Ki, Kd, previous_error, integral):
    error = setpoint - current_value
    integral += error
    derivative = error - previous_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, error, integral

def reVel():
    return current_vel.twist.linear.x, current_vel.twist.linear.y, current_vel.twist.linear.z

def reThunder():
    return current_thunder[0], current_thunder[1]

def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) 
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def reset_gazebo_simulation():
    '''rospy.wait_for_service('/gazebo/reset_world')
    try:
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        rospy.loginfo("Gazebo world has been reset to its initial state.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    '''
    model_state_pub = rospy.Publisher('/gazebo/set_model_states', ModelStates, queue_size=1)
    poses_msg = ModelStates()
    poses_msg.name = [None]
    poses_msg.pose = [Pose()]
    poses_msg.twist = [Twist()]
    
    poses_msg.name[0] = 'target_true'
    
    radius = random.uniform(20, 30)
    angle = random.uniform(0, 2 * math.pi)

         
    poses_msg.pose[0].position.x = radius * math.cos(angle)
    poses_msg.pose[0].position.y = radius * math.sin(angle)
    model_state_pub.publish(poses_msg)

