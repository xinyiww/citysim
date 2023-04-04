#!/usr/bin/python

"""
This visualization module is for generating generalized grid-world from different lane geometries, including T-junc and lane change. 
Dec 12 
Run successfully on rosbag
Dec 13 Adding traffic features.
Dec 15 Add lane snapping to assign oppo cars onto each lane, could be run seperately on real time carla
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import rospy
import rosnode
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path, Odometry
from math_utils import quaternion_to_euler

from traffic_msgs.msg import Waypoint, WaypointArray, VehicleStateArray, VehicleState, PerceptionLanes, IntentionTrajectory, Prediction, PredictionArray, PredictionLanes, CenterLanes, PedestrianStateArray, PedestrianState
from tf2_msgs.msg import TFMessage
from std_msgs.msg import UInt8


import matplotlib.patches as patches
import matplotlib.colors as mcolors

import sys
# sys.path.append("/home/xliu/Documents/ros_record")
from  bag_from_citysim import *
np.set_printoptions(precision=4)

VISUALIZE_PATH = 0
VISUALIZE_SCENE = 0
VISUALIZE_TRAFFIC = 1
VISUALIZE_VELOCITY = 1
VISUALIZE_TRAFFIC_GRIDWORLD = 0

VISUALIZE_LANES = 0
VISUALIZE_LANES_CENTER = 1
VISUALIZE_LANES_GRIDWORLD = 0
VISUALIZE_PREDICTION = 1
VISUALIZE_GT_AGAINST_PREDICTION = 1
LOG_ALL = 0
EGO_STATE = 0


target_lane_id = 1
color = 'rgb'
color_list = ['red', 'green', 'blue','orange','brown', 'pink','yellow', 'purple','aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque']

lane_fn = "dataset/McCulloch@SeminoleLanes.npy"
gt_fn = "dataset/McCulloch@Seminole-01.csv"
pred_topic_name = '/region/all_cars_predictions_GM'
class Visualize_Interface():
    def __init__(self):

        rospy.init_node("Visualize_pred", anonymous=False)
        if VISUALIZE_TRAFFIC:
            rospy.Subscriber('/region/lanes_perception', PerceptionLanes, self.vehicle_callback)
        if VISUALIZE_LANES_CENTER: 
            rospy.Subscriber('/region/lanes_center', CenterLanes, self.lane_callback)
        if EGO_STATE: 
            rospy.Subscriber('/region/ego_state', VehicleState, self.ego_callback)
        rospy.Subscriber('/region/global_route', Path, self.route_callback)  # ROUTE PLANNER
        if VISUALIZE_PREDICTION:
            rospy.Subscriber(pred_topic_name, PredictionArray, self.all_preds_callback)


        rospy.loginfo("Waiting for traffic...")
        try:
            if VISUALIZE_TRAFFIC:
                rospy.wait_for_message('/region/lanes_perception', PerceptionLanes,timeout=10.0)
            if VISUALIZE_LANES_CENTER: 
                rospy.wait_for_message('/region/lanes_center', CenterLanes,timeout=10.0)
            # if EGO_STATE:   
            #     rospy.wait_for_message("/region/ego_state", VehicleState, timeout=10.0)
            if VISUALIZE_PREDICTION:
                rospy.wait_for_message(pred_topic_name, PredictionArray, timeout=10.0)
        except rospy.ROSException as e:
            rospy.logerr("Timeout while waiting for traffic info!")
            raise e
        rospy.loginfo("Done waiting")
        if VISUALIZE_GT_AGAINST_PREDICTION:
            self.GT = getGTData(gt_fn)
        try:
            rate = rospy.Rate(20)  # 10hz
            check_id = 0
            ct = 0
            
            
            while not rospy.is_shutdown():
                
                fig1 = plt.figure(1)
                
                fig1.clf()
                ax1 = fig1.add_subplot(111, aspect='equal') 
                ax1.set_xlim(0,160)
                ax1.set_ylim(0,100)
                # ax2 = fig1.add_subplot(212)
                # ax1.set_xlim([88,102])
                # # ax2.set_xlim([0,10])
                # ax2.set_xlabel("horizon t (s)")
                # ax2.set_ylabel("longitute related prediction (m)")
                
                
        

                if VISUALIZE_SCENE:
                    
                    ax1.plot(self.global_path[:,0], self.global_path[:,1],'.b', label = "global_path")
                    ax1.plot(self.ego_pose_carla[0], self.ego_pose_carla[1],'or', label = "ego_pose_carla")
                    ax1.arrow(self.ego_pose_carla[0] , self.ego_pose_carla[1], 
                              dx= np.cos(yaw) * self.ego_vel_carla, dy= np.sin(yaw) * self.ego_vel_carla, head_width=1)
                    
                
                if  VISUALIZE_LANES:    
                    # for i, lane in enumerate(self.lane_all):
                    #     ax1.plot(lane[0], lane[1], '-'+color[i], label = "lane_"+str(i))
                    # fn = 'RoundaboutALane.npy'
                    all_lane = np.load(lane_fn, allow_pickle=True)
                    for i in range(all_lane.shape[0]):
                        xys = all_lane[i].reshape((all_lane[i].shape[0], 2))* 0.128070 * 0.3048
                        ax1.plot(xys[:,0], xys[:,1], c = 'grey',
                                alpha = 0.3)
                
                if  VISUALIZE_LANES_CENTER:    
                    for i, lane in enumerate(self.lane_all):
                        ax1.plot(lane[0], lane[1], 
                                #  '-'+color_list[i], 
                                # label = "lc_"+str(self.lanes_ids[i])
                                ) 
                        ax1.text(lane[0,0], lane[1,0], "lc_"+str(self.lanes_ids[i]))   
                        # ax1.legend()    
                
                if VISUALIZE_PREDICTION:
                    ct +=1 
                    new_veh_ids = []
                    for i, lane in enumerate(self.traffic_data.vehicles):
                        for j, veh in enumerate(lane.vehicles):
                            new_veh_ids.append(veh.lifetime_id)
                            
          
                            
                    for i, veh in enumerate(self.all_pred_data.predictions):
                        # if veh.agent_id == check_id:
                        if True:
                            xs = []
                            ys = []
                            
                            t_start = self.all_pred_data.header.stamp.to_sec()
                            ts = t_start + np.linspace(0, 10, 101)
                            n_pred = len(veh.trajectories[0].trajectory_uncertainty.waypoints) # assume we only have one discrete possibility
                            dt = veh.dt
                            
                                  
                                
                            for j, wps in enumerate(veh.trajectories[0].trajectory_estimated.waypoints):
                                x = wps.pose.pose.position.x
                                y = wps.pose.pose.position.y
                                
                                xs.append(x)
                                ys.append(y)
                                
                            node_names = rosnode.get_node_names()
                            for node_name in node_names:
                                pre = '/Prediction_'
                                n_pre = len(pre)
                                if node_name[:n_pre] == pre:
                                    label_name = node_name[n_pre:]
                            ax1.scatter(np.array(xs), np.array(ys), marker = "*", c = 'blue',alpha=0.3) 
                            # ax1.text(np.array(xs)[-1], np.array(ys)[-1], str(veh.agent_id))
                            # ax2.scatter(np.array(ts)[:n_pred], np.array(ys),label = label_name)

                        
                            if VISUALIZE_GT_AGAINST_PREDICTION:
                                # adding gt visualization
                                tixys_gt = self.GT[self.GT[:,1] == veh.agent_id]
                                # ax1.scatter(tixys_gt[:,2], tixys_gt[:,3], marker = "o", c = 'yellow',alpha=0.1)
                                xs_gt = np.interp(ts, tixys_gt[:,0], tixys_gt[:,2], right = np.nan)
                                ys_gt = np.interp(ts, tixys_gt[:,0], tixys_gt[:,3], right = np.nan)
                                ax1.scatter(xs_gt, ys_gt, marker = "o", c = 'yellow',alpha=0.3)
                        ax1.legend()
                        # ax2.legend()
                       
                              
                if VISUALIZE_TRAFFIC:
                    
                    for i, lane in enumerate(self.traffic_data.vehicles):
                        
                        for j, veh in enumerate(lane.vehicles):
                            
                        
                            o = veh.pose.pose.orientation
                            yaw, pitch, roll = quaternion_to_euler(o.x, o.y, o.z, o.w)             
                            pos = np.asarray([[veh.pose.pose.position.x-1/2* veh.length*np.cos(yaw), veh.pose.pose.position.y-1/2* veh.length*np.sin(yaw)],
                                                [veh.pose.pose.position.x+1/2* veh.length*np.cos(yaw), veh.pose.pose.position.y+1/2*veh.length*np.sin(yaw)]])
                            ax1.plot(pos[:,0], pos[:,1],'b', linewidth=7, label = str(veh.lifetime_id)) 
                            vx,vy = veh.twist.twist.linear.x, veh.twist.twist.linear.y
                            ax1.text (veh.pose.pose.position.x, veh.pose.pose.position.y + 5, 
                                     "id="+str(veh.lifetime_id)+", vel="+ '{:.1f}'.format(np.sqrt(vx**2 + vy**2)))
                            
                            if VISUALIZE_VELOCITY:
                                arrow_start = (veh.pose.pose.position.x, veh.pose.pose.position.y)
                                arrow_length = (veh.twist.twist.linear.x, veh.twist.twist.linear.y)
                                ax1.arrow(*arrow_start, *arrow_length, color='red', width=1)
                    
                # ax1.legend()
                plt.show(block=False)
                plt.pause(0.0000000001)
                

                rate.sleep()


            rospy.spin()

        finally:
            rospy.loginfo("Done")

    def vehicle_callback(self, data):
        self.traffic_data = data
        


    # def pedestrian_callback(self, data):
    #     self.pedestrian_data = data

    def lane_callback(self, data):
        self.lane_data = data
        self.lane_all = []
        self.lanes_ids = self.lane_data.ids 
        for i in range(len(self.lane_data.center_lines)):
            l = self.lane_data.center_lines[i]
            xys = np.array([
                [l_pose.pose.position.x for l_pose in l.path.poses],
                [l_pose.pose.position.y for l_pose in l.path.poses]
                ])
            self.lane_all.append(xys)
            

        
    def lane_snap(self, x,y):
        dist_to_diff_lane = np.ones(len(self.lane_all))*100000
        for i, lane in enumerate(self.lane_all):
            dists = np.linalg.norm(np.array([[x],[y]]) -  self.lane_all[i], axis = 0)
            dist_to_diff_lane[i] = np.min(dists)
        
        return np.argmin(dist_to_diff_lane)    
            
        

    def ego_callback(self, data):
        self.ego_data_set = 1
        self.ego_data = deepcopy(data)

    def all_preds_callback(self, data):
        self.all_pred_data = data


    def ped_preds_callback(self, data):
        self.ped_pred_data = data

    def route_callback(self,data):
        self.global_route = data

def angle_wrap_0_2pi_from_np(angle_radian):
    if angle_radian < 0: 
        return angle_wrap_0_2pi_from_np(angle_radian + np.pi * 2)
    elif angle_radian >= 2 * np.pi:
        return angle_wrap_0_2pi_from_np(angle_radian - np.pi * 2)
    else:
        return angle_radian 

if __name__ == "__main__":
	try:
		Visualize_Interface()
	except rospy.ROSInterruptException:
		pass