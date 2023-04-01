
import datetime
import numpy as np
import math
import tf
from traffic_msgs.msg import CenterLanes, PerceptionLanes, WaypointArray, Waypoint, VehicleStateArray, VehicleState, PathWithSpeed
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
import pandas as pd
import rospy
from math_utils import *
import rosbag
from scipy.interpolate import interp1d


def getGTData(filename):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    column_names = ['frameNum', 
                    'carId', 
                    'carCenterXft', 
                    'carCenterYft',
                    ]
    for col in column_names:
        globals()[col] = np.array(df[col].tolist()) 
    
    data = np.zeros([len(frameNum), 4])
            # data[:, 0] = t, veh_id, x, y
    data[:, 0], data[:, 1], data[:, 2], data[:, 3] = frameNum/30, carId, carCenterXft* 0.3048, carCenterYft* 0.3048
    return data


def calculate_velocity(xs, ys, ts, idx, n_f, n_b):
    # xs, ys, ts: traj in x, y axis
    # idx: idx that we want to estimate the velocity
    # n_f, n_b: data points at front, at back (eg. idx -n_f |____|____| idx + n_f)
    # Interpolate the data using a quadratic spline
    tck_x = interp1d(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], 
                     xs[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], kind='cubic', bounds_error=False, fill_value="extrapolate")
    tck_y = interp1d(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], 
                     ys[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Differentiate the splines to obtain the velocity in the x and y directions
    vxs = np.gradient(tck_x(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)]), 1/30)
    vys = np.gradient(tck_y(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)]), 1/30)
    vx, vy= vxs[min(n_f, idx)], vys[min(n_f, idx)]
    return vx,vy



def extract_length_width(p1, p2, p3):
    # Identify the opposite corners of the rectangle
    diag1 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    diag2 = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
    diag3 = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
    # Calculate the width and length of the rectangle
    width, length, _ = np.sort([diag1, diag2, diag3])
    return length, width

#extract_center_lane from averaged trajectories
def write_msg_center_lanes(Ts, laneId, carId, carCenterX, carCenterY):
    msg_cl = CenterLanes()
    lane_ids = (id for id in np.unique(laneId))
    for l in lane_ids: 
        veh_ids = (id for id in np.unique(carId[laneId == l]))
        traj_list = []
        for id in veh_ids:
            mask1 = carId == id 
            mask2 = laneId == l
            ts, xs, ys = Ts[mask1 & mask2], carCenterX[mask1 & mask2], carCenterY[mask1 & mask2]
            
            if l == 4 and ys[-1]-ys[0] < 0:
                xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), np.flip(xs)), 
                             np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), np.flip(ys)))
            else:
                xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), xs), 
                             np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), ys))
            traj_list.append([xs_int, ys_int]) 
        traj_list = np.array(traj_list)
        xs_avg, ys_avg = np.mean(traj_list[:,0,:], axis=0), np.mean(traj_list[:,1,:], axis=0)
        
        cl = PathWithSpeed()
        for i in range(xs_avg.shape[0]):
            pt = PoseStamped()
            pt.pose.position.x, pt.pose.position.y = xs_avg[i], ys_avg[i]
            cl.path.poses.append(pt)
 
        # fill in msg
        msg_cl.ids.append(l)
        msg_cl.center_lines.append(cl)

    return msg_cl
def visualize_problem(Ts, laneId, carId, carCenterX, carCenterY):
    lane_ids = [id for id in np.unique(laneId)]
    for l in lane_ids: 
        traj_list = []
        veh_ids = [id for id in np.unique(carId[laneId == l])]
        for id in veh_ids:
            mask1 = carId == id 
            mask2 = laneId == l
            ts, xs, ys = Ts[mask1 & mask2], carCenterX[mask1 & mask2], carCenterY[mask1 & mask2]
            if l == 4 and ys[-1]-ys[0] < 0:
                xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), np.flip(xs)), 
                             np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), np.flip(ys)))
            else:
                xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), xs), 
                             np.interp(np.linspace(0,1,100), (ts - ts[0])/(ts[-1] - ts[0]), ys))
            traj_list.append([xs_int, ys_int]) 
        traj_list = np.array(traj_list)
        xs_avg, ys_avg = np.mean(traj_list[:,0,:], axis=0), np.mean(traj_list[:,1,:], axis=0)
        for xys in traj_list:
            plt.plot(xys[0], xys[1], lw = 1, c='grey')
        plt.plot(xs_avg, ys_avg, lw = 5, c='red',  label= 'lane_id ='+str(l))
        plt.scatter(xs_avg[0], ys_avg[0], s=40,c='blue', label='starting point')
        plt.legend()
        plt.show()
        # xs_std, ys_std = np.std(traj_list[:,0,:], axis=0), np.std(traj_list[:,1,:], axis=0)
        # ds_std = np.sqrt(xs_std * xs_std + ys_std * ys_std)
        
def check_rel_difference(vx,vy, speed):
    print("calculated speed = ", np.sqrt(vx**2 + vy ** 2), "speed from raw data (m/s)", speed)

def write_msg_lane_perc(t,Ts,indices, laneId, carId, carCenterX, carCenterY, boundingBox1X, boundingBox1Y, boundingBox2X, boundingBox2Y, boundingBox3X, boundingBox3Y, speed):
    msg = PerceptionLanes()
    timestamp = rospy.Time(t)
    header = Header()
    header.stamp = timestamp
    msg.header = header
    # ids:
    lane_ids_cur = laneId[indices]
    msg.ids = list(id for id in np.unique(lane_ids_cur))
    # Vehicle information
    for l in msg.ids:
        lane_vehs = VehicleStateArray()
        
        for i in indices:
            if laneId[i] == l: 
                # build a vehicle state instance
                veh = VehicleState()
                veh.lifetime_id = carId[i]
                p1, p2, p3 = (boundingBox1X[i], boundingBox1Y[i]), (boundingBox2X[i], boundingBox2Y[i]), (boundingBox3X[i], boundingBox3Y[i])
                veh.length, veh.width = extract_length_width(p1, p2, p3)
                veh.pose.pose.position.x, veh.pose.pose.position.y = carCenterX[i], carCenterY[i]
                veh.pose.pose.orientation.x, veh.pose.pose.orientation.y, veh.pose.pose.orientation.z, veh.pose.pose.orientation.w = tf.transformations.quaternion_from_euler(0.0, 0.0, heading[i]/180 * np.pi + np.pi/2) 
                # we have to calculate an estimated velocity by hand
                xs, ys, ts = carCenterX[carId == carId[i]], carCenterY[carId == carId[i]], Ts[carId == carId[i]]
                idx = np.where(ts == t)[0][0]
                vx, vy = calculate_velocity(xs, ys, ts,idx, 8, 8)
                # check_rel_difference(vx,vy,speed[i])
                veh.twist.twist.linear.x, veh.twist.twist.linear.y = vx, vy
                lane_vehs.vehicles.append(veh)
            
        msg.vehicles.append(lane_vehs)
    return  msg

def build_traffic_bag_from_data(data_csv_fn, output_bag_fn):
    
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(data_csv_fn)
    
    # Store all column names in a list
    column_names = ['frameNum', 
                    'carId', 
                    'carCenterXft', 
                    'carCenterYft',
                    'boundingBox1Xft', 
                    'boundingBox1Yft',
                    'boundingBox2Xft', 
                    'boundingBox2Yft',
                    'boundingBox3Xft', 
                    'boundingBox3Yft',
                    'speed', 
                    'heading',  
                    'laneId']
    
    # Create variables with the same names as columns and assign the column data to them
    for col in column_names:
        globals()[col] = np.array(df[col].tolist())
    freq = 30
    ts = frameNum / freq
    
    
    bag = rosbag.Bag(output_bag_fn, 'w')
       
    try:
        # visualize_problem(ts, laneId, carId, carCenterX, carCenterY)
        msg_cl = write_msg_center_lanes(ts, laneId, carId, carCenterXft * 0.3048, carCenterYft * 0.3048)

        print("# frames = ", np.unique(ts))
        for t in np.unique(ts):
            # print(ts[ts == t].shape[0])
            indices = np.where(ts == t)[0]
            # make sure all input are in meters and seconds
            
            msg_lp = write_msg_lane_perc(t, ts, indices, laneId, carId, 
                                carCenterXft * 0.3048, carCenterYft * 0.3048, 
                                boundingBox1Xft * 0.3048, boundingBox1Yft * 0.3048, boundingBox2Xft * 0.3048, boundingBox2Yft * 0.3048, boundingBox3Xft * 0.3048, boundingBox3Yft * 0.3048, speed* 0.44704)

            # bag.write('/region/lanes_center', msg_cl_filtered, rospy.Time(t))
            bag.write('/region/lanes_center', msg_cl, rospy.Time(t))
            bag.write('/region/lanes_perception', msg_lp, rospy.Time(t))
            print("msg sent at t = ", t)
            if t >= 50:
                break      
    finally:
        bag.close()

if __name__ == "__main__":
    # input_csv_fn = 'RoundaboutA-02.csv'
    input_csv_fn = "bags/McCulloch@Seminole-01.csv"
    output_bag_fn = input_csv_fn[:-4]+".bag"
    build_traffic_bag_from_data(input_csv_fn, output_bag_fn)
        
    