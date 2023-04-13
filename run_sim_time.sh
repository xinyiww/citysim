#!/bin/bash
ROSBAG=bags/McCulloch_gt_use_both.bag
# ROSBAG=bags/RoundaboutA_gt_use_path.bag
roscore & 
rosparam set use_sim_time true # using the simulated time
rosbag play -l -r 0.2 -s 25 $ROSBAG --clock &
rosrun prediction_ct_vel prediction_ct_vel_node & 
# rosrun prediction_GM prediction_GM_node & 
# rosrun prediction_IDM prediction_IDM_node & 
# rosrun prediction_combo prediction_combo_node &
python3 rosnode_visualization.py

### use rosnode kill -a if bag still runs after you kill the script