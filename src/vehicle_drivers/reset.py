import sys
import os
import argparse
from math import pi

import numpy as np

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from ackermann_msgs.msg import AckermannDrive

ackermann_msg = AckermannDrive()
ackermann_msg.steering_angle_velocity = 0.0
ackermann_msg.acceleration = 0.0
ackermann_msg.jerk = 0.0
ackermann_msg.speed = 0.0
ackermann_msg.steering_angle = 0.0

def getModelState():
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        modelState = serviceResponse(model_name='gem')
    except rospy.ServiceException as exc:
        rospy.loginfo("Service did not process request: " + str(exc))
    return modelState

def setModelState(model_state):
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(model_state)
    except rospy.ServiceException as e:
        rospy.loginfo("Service did not process request: " + str(e))

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def set_position(x=0, y=0, yaw=0):
    rospy.init_node("set_pos")

    curr_state = getModelState()
    new_state = ModelState()
    new_state.model_name = 'gem'
    new_state.pose = curr_state.pose
    new_state.pose.position.x = x
    new_state.pose.position.y = y
    new_state.pose.position.z = 1
    q = euler_to_quaternion([yaw, 0, 0])
    new_state.pose.orientation.x = q[0]
    new_state.pose.orientation.y = q[1]
    new_state.pose.orientation.z = q[2]
    new_state.pose.orientation.w = q[3]
    new_state.twist = curr_state.twist
    setModelState(new_state)
    ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
    ackermann_pub.publish(ackermann_msg)

if __name__ == "__main__":

    #run "python3 reset.py --config track" for track, and "...parking" for parking
    
    parser = argparse.ArgumentParser(description='Set the vehicle position based on predefined configurations.')

    configurations = {
        'track': {'x': -5, 'y': -21, 'yaw': pi},
        'parking': {'x': -40, 'y': 23, 'yaw': 0},
        # Add more configurations here
    }

    parser.add_argument('--config', type=str, help='Configuration name.', choices=configurations.keys())

    argv = parser.parse_args()

    config = configurations.get(argv.config, configurations['track'])  # Default to 'track' if no config is specified

    set_position(x=config['x'], y=config['y'], yaw=config['yaw'])
