#!/bin/bash
roscore & 
rosparam set use_sim_time true
rosbag play -l bags/McCulloch@Seminole-01.bag --clock &
rosrun prediction_ct_vel prediction_ct_vel_node & 
# python3 rosnode_visualization.py