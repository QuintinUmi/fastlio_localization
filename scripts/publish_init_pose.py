#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

def get_param_vector(name, default):
    vals = rospy.get_param(name, default)
    # 支持参数为 list/string，两者都转成 float 数组
    if isinstance(vals, str):
        vals = [float(x) for x in vals.strip().split()]
    return [float(x) for x in vals]

if __name__ == "__main__":
    rospy.init_node("ndt_init_pose_pub")
    pub = rospy.Publisher("/initpose", Odometry, queue_size=1, latch=True)
    rospy.sleep(1.0)

    # 默认值可以适当调整
    pos = get_param_vector("init_pose/position", [0.0, 0.0, 0.0])
    ori = get_param_vector("init_pose/orientation", [0.0, 0.0, 0.0, 1.0])

    msg = Odometry()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.child_frame_id = "base_link"
    msg.pose.pose.position = Point(pos[0], pos[1], pos[2])
    msg.pose.pose.orientation = Quaternion(ori[0], ori[1], ori[2], ori[3])
    pub.publish(msg)
    rospy.loginfo("Published initial odometry pose: pos=%s ori=%s", pos, ori)
    rospy.spin()