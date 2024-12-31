
import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import GetLinkState

def pose_publisher():
    relative_pose_pub = rospy.Publisher("/gazebo/zhihang/standard_vtol/local_position", Pose, queue_size = 1)

    f = 10.0
    rate = rospy.Rate(f)
    while not rospy.is_shutdown():
        try:
            response = get_link_state('standard_vtol_0::base_link', 'ground_plane::link')
            relative_pose = response.link_state.pose

            relative_pose_pub.publish(relative_pose)
        except:
            continue
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('zhihang1')
    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    pose_publisher()
