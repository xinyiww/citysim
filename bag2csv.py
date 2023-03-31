#!/usr/bin/env python

import os
import csv
import rosbag
import sys

import numpy as np

def bag2csv(filename, directory, isGT):
  # Read the rosbag file
  print("Reading the rosbag file")
  if not directory.endswith("/"):
    directory += "/"
  extension = ""
  if not filename.endswith(".bag"):
    extension = ".bag"
  bag = rosbag.Bag(directory + filename + extension)

  # Create directory with name filename (without extension)
  results_dir = directory + filename[:] + "_results"
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
    
  if not isGT:
    print("Writing prediction data")
    
    pred_topics = set()
    for topic, msg, t in bag.read_messages():
      if topic[:28] == '/region/all_cars_predictions':
        pred_topics.add(topic)
    print(pred_topics)
    for tp in pred_topics:
      if len(tp) == 28:
        
        fn = 'pred_ct_vel'
      else:
        fn = 'pred'+ tp[28:]
        
      with open(results_dir +"/"+fn+'.csv', 'a', newline='') as pred_csv:
        log_writer = csv.writer(pred_csv, delimiter=' ')      
        
        
        for topic, msg, t in bag.read_messages(topics=[tp]):
          if msg.predictions != []:
            for i, veh in enumerate(msg.predictions):
            
              xs = []
              ys = []
              dxs = []
              dys = []

              t_start = msg.header.stamp.to_sec()
              # print(t_start)
              n_pred = len(veh.trajectories[0].trajectory_uncertainty.waypoints) # assume we only have one discrete possibility
              dt = veh.dt      
                                  
              for j, wps in enumerate(veh.trajectories[0].trajectory_estimated.waypoints):
                x = wps.pose.pose.position.x
                y = wps.pose.pose.position.y
                
                xs.append(x)
                ys.append(y)
            
              
                  
              for j, wps in enumerate(veh.trajectories[0].trajectory_uncertainty.waypoints):
                dx = wps.pose.pose.position.x
                dy = wps.pose.pose.position.y
                
                dxs.append(dx)
                dys.append(dy) 
                
              veh_id = veh.agent_id 
                  
              log_writer.writerow([t_start,dt,n_pred,veh_id] + xs + ys + dxs + dys )
  
  else: 
    print("Writing gt data")
    fn = 'gt.csv'
    with open(results_dir +"/"+fn, 'a', newline='') as gt_csv:
        log_writer = csv.writer(gt_csv, delimiter=' ')   
        for topic, msg, t in bag.read_messages(topics=['/region/lanes_perception']):
            t = msg.header.stamp.to_sec()

            for i, lane in enumerate(msg.vehicles):            
                for j, veh in enumerate(lane.vehicles): # some deplicates in the first round -- delete them
                    pos = veh.pose.pose.position   
                    veh_id = veh.lifetime_id
                    log_writer.writerow([t, veh_id, pos.x, pos.y] )


if __name__ == "__main__":

  fn_PRED = sys.argv[1]
  directory = sys.argv[2]

  bag2csv(fn_PRED, directory, isGT = True)
  bag2csv(fn_PRED, directory, isGT = False)


      
      
