#!/bin/bash
# roscore & 
rosparam set use_sim_time true
rosbag play -l -r 0.3 -s 35 bags/McCulloch_gt.bag --clock &
rosrun prediction_GM prediction_GM_node & 
# rosrun prediction_ct_vel prediction_ct_vel_node & 
python3 rosnode_visualization.py