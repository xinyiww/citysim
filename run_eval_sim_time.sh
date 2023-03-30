#!/bin/bash

# where I store those random parameter sets in 125 * 3 sheet
CSV_FILE=params_random_search_lc.csv
# define the rosbag we gain GT and generate prediction
ROSBAG=/home/xliu/Documents/ros_record/random_search/lane_change_pred_1_filtered.bag # should not contain /region/lanes_prediction and /region/all_cars_predictions_idm
# working space
WS=/home/xliu/Documents/ros_record/random_search/
GT=gt_lc_1.csv
# define GT
# python generate_random_sets.py $CSV_FILE $WS # only use once for each trial
python bag2csv_local.py $ROSBAG $WS $GT

roslaunch prediction_IDM run_idm_node.launch &
# rosrun prediction_IDM prediction_IDM_node &
# Skip the header row
tail -n +2 "$CSV_FILE" | while IFS=, read -r id max_acc min_d_gap min_t_gap target_speed; do 
  echo "max_acc: $max_acc , min_d_gap: $min_d_gap , min_t_gap: $min_t_gap target_speed: $target_speed"
  rosparam set /max_acc $max_acc
  rosparam set /min_d_gap $min_d_gap
  rosparam set /min_t_gap $min_t_gap
  rosparam set /target_speed $target_speed
  # rosparam set /max_deacc $max_deacc
  # rosparam set /quantitative_accuracy $quantitative_accuracy
  rosparam set use_sim_time true
  # Start recording messages to a rosbag file
  rosbag record -e "/region/all_cars_predictions_idm" -O params_pred_${id}.bag __name:=my_bag & </dev/null
  # Play the rosbag file
  rosbag play $ROSBAG --clock </dev/null

  # Kill the ROS nodes and stop recording messages
  rosnode kill my_bag
  # rosnode cleanup

done

python eval_and_save.py  $GT
# ros_pid=$(pgrep roscore)
# kill -SIGTERM $ros_pid

