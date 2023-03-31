#!/bin/bash
# roscore & 
rosparam set use_sim_time true
rosbag play -l -s 30 bags/McCulloch_gt.bag --clock &
rosrun prediction_ct_vel prediction_ct_vel_node & 
python3 rosnode_visualization.py