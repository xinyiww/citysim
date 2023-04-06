#!/bin/bash
# ROSBAG=bags/McCulloch_pred_use_path.bag
ROSBAG=bags/RoundaboutA_gt_use_path.bag
# roscore & 
rosparam set use_sim_time true
rosbag play -l -r 0.2 -s 25 $ROSBAG --clock &
rosrun prediction_GM prediction_GM_node & 
# rosrun prediction_ct_vel prediction_ct_vel_node & 
python3 rosnode_visualization.py