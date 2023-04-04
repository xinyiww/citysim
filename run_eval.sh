#!/bin/bash

# define the rosbag we gain GT and generate prediction
BAG_GT=bags/McCulloch_gt_use_path # working space
BAG_PRED=bags/McCulloch_pred_use_path
WS=/home/xliu/Documents/dataset/CitySim/
GT=dataset/McCulloch@Seminole-01.csv
rosparam set use_sim_time true

# roscore & 

rosrun prediction_ct_vel prediction_ct_vel_node & 
rosrun prediction_GM prediction_GM_node &
rosbag record -a -O ${BAG_PRED}.bag __name:=my_bag & </dev/null

rosbag play -r 1 ${BAG_GT}.bag --clock </dev/null
# Kill the ROS nodes and stop recording messages
rosnode kill my_bag
# rosnode cleanup

python3 bag2csv.py $BAG_PRED $WS
python3  eval.py ${BAG_PRED}_results/
# ros_pid=$(pgrep roscore)
# kill -SIGTERM $ros_pid

