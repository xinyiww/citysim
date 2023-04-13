
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
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from plot_utils import *

# this python file generate ros bags from citysim dataset. It is also called a citysim interface 
# including the following topics:
#    - /region/lanes_center
#    - /region/lanes_perception

# scaling:
#  - unsignalized intesection(dataset/McCulloch@Seminole-01.csv): to_ft: 0.128070 to_meters: to_ft * 0.3048
#  - RoundaboutA-02 to_ft: 0.213267 to_meters: to_ft * 0.3048

# xinyi (updated on Apr 11) lane_id was set to l*MAX_NUM_LANES + i if there are multiple possible paths in one lane section

# the mininum apearance of a car in frames, 30 means a second
MIN_APPEAR_TIME = 30
# the transformation from foot to meter
FT_TO_METER = 0.3048
# the maximum number of lanes
# MAX_NUM_LANES = 100
DEG_TO_RAD = np.pi / 180
 
# plot lane boundary
def getLaneRaw(lane_fn):
    fig1 = plt.figure(1)
                
    fig1.clf()
    ax1 = fig1.add_subplot(111, aspect='equal') 

    all_lane = np.load(lane_fn, allow_pickle=True)
    for i in range(all_lane.shape[0]):
        xys = all_lane[i].reshape((all_lane[i].shape[0], 2))
        l = xys.shape[0]
        ax1.plot(xys[:,0], xys[:,1], c = 'grey',
                alpha = 0.3)
        ax1.text(xys[0,0], xys[0,1], "lc_"+str(i)) 
    plt.show()


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
    data[:, 0], data[:, 1], data[:, 2], data[:, 3] = frameNum/30, carId, carCenterXft* FT_TO_METER, carCenterYft* FT_TO_METER
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

# using hand-assigned lane center
def hand_assign_lane_center_interface(npy_fn, scale_pixal_to_meters, load = True):
    msg_cl = CenterLanes()
    all_lane = np.load(npy_fn, allow_pickle=True)
    lane_ids = range(all_lane.shape[0])
    if not load:
        with open(npy_fn[:-4]+"_lane_center.npy", 'wb') as f:
            for l in lane_ids:
                xs, ys, is_used = assign_points_interface(npy_fn, l, scale_pixal_to_meters)
                np.save(f, l)
                np.save(f, xs)
                np.save(f, ys)  
                
                cl = PathWithSpeed()
                for i in range(xs.shape[0]):
                    pt = PoseStamped()
                    pt.pose.position.x, pt.pose.position.y = xs[i], ys[i]
                    cl.path.poses.append(pt)
        
                # fill in msg
                msg_cl.ids.append(l)
                msg_cl.center_lines.append(cl)
                    
    else: 
        with open(npy_fn[:-4]+"_lane_center.npy", 'rb') as f:
            for _ in range(len(lane_ids)):
                l = np.load(f)
                xs = np.load(f)
                ys = np.load(f)
                
                cl = PathWithSpeed()
                for i in range(xs.shape[0]):
                    pt = PoseStamped()
                    pt.pose.position.x, pt.pose.position.y = xs[i], ys[i]
                    cl.path.poses.append(pt)
        
                # fill in msg
                msg_cl.ids.append(l)
                msg_cl.center_lines.append(cl)
            
        

    return msg_cl

# adjacent maps of land ids
def compute_connectivity_representation(carId,laneId):
    veh_ids_all = np.unique(carId)
    # lane_ids = np.unique(laneId)
    lane_ids = np.arange(max(laneId)+1)
    # init adjacent matrix and map_car_lanes
    adj_mtr = np.zeros([len(lane_ids), len(lane_ids)])
    map_car_lanes = {} # car ids -> lanes_passed seq
    for cid in veh_ids_all:
        car_in_lanes = laneId[carId == cid]
        lane_ids_car_v, lane_ids_car_i = np.unique(car_in_lanes, return_index=True)
        lane_ids_car = lane_ids_car_v[np.argsort(lane_ids_car_i)]
        map_car_lanes[cid] = tuple(lane_ids_car)
        for i in range(lane_ids_car.shape[0] -1):
            adj_mtr[lane_ids_car[i], lane_ids_car[i+1]] = 1
    return adj_mtr, map_car_lanes

def find_sets_containing_element(lst, element):
    return [s for s in lst if element in s]



# let's define path to be a 
def compute_paths(carId, carCenterXft, carCenterYft, laneId, visualize_paths = True):
    veh_ids_all = np.unique(carId)
    # lane_ids = np.unique(laneId)
    lane_ids_old = np.arange(max(laneId)+1)
    _, map_car_lanes =compute_connectivity_representation(carId,laneId)
    
    cases = list(set(np.array(list(map_car_lanes.values()))))
    cases = sorted(cases, key=lambda x: ( len(x), x[0],x[len(x)-1])) # starting lane, ending lane, crossing lane
    
    msg_path = CenterLanes()
    path_ids = range(len(cases))
    
    # build a map from car_id to path_id
    map_car_path = {}
    
    for path_id in path_ids:
        case = cases[path_id]
        cars = [car_id for car_id, lanes in map_car_lanes.items() if lanes == case]
        traj_list = []
        
        for car_id in cars: 
            # print(len(carCenterXft[carId == car_id]) >= MIN_APPEAR_TIME)
            # if len(carCenterXft[carId == carId[i]]) >= MIN_APPEAR_TIME:
            if len(carCenterXft[carId == car_id]) <= MIN_APPEAR_TIME:
                print("car_id = ", car_id)
            if True:
                map_car_path[car_id] = path_id
                xs = carCenterXft [carId == car_id]* 0.3048
                ys = carCenterYft [carId == car_id]* 0.3048
                ds = np.cumsum(np.sqrt(np.diff(xs,prepend=xs[0])**2 + np.diff(ys,prepend=xs[0])**2))
                xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ds - ds[0])/(ds[-1] - ds[0]), xs), 
                                np.interp(np.linspace(0,1,100), (ds - ds[0])/(ds[-1] - ds[0]), ys))
                traj_list.append([xs_int, ys_int]) 
        traj_list = np.array(traj_list)
        if traj_list.shape[0] != 0:
            xs_avg, ys_avg = np.mean(traj_list[:,0,:], axis=0), np.mean(traj_list[:,1,:], axis=0)
            xs_sm, ys_sm = savgol_filter(xs_avg, 21, 3),savgol_filter(ys_avg, 21, 3)
            if visualize_paths:
                for xys in traj_list:
                    plt.plot(xys[0], xys[1], lw = 1, c='grey')
                # plt.plot(xs_avg, ys_avg, lw = 5, c='red',  label= 'lane_id ='+str(path_id))
                plt.plot(xs_sm, ys_sm, lw = 2, c='green',  label= 'avg path of path_id ='+str(path_id))
                plt.scatter(xs_sm[0], ys_sm[0], s=40,c='blue', label='starting point')
                plt.legend()
                plt.xlabel(" x position (m)")
                plt.ylabel(" y position (m)")
                plt.axis("equal")
                plt.show()
                    
        
            cl = PathWithSpeed()
            for i in range(xs_sm.shape[0]):
                pt = PoseStamped()
                pt.pose.position.x, pt.pose.position.y = xs_sm[i], ys_sm[i]
                cl.path.poses.append(pt)
    
            # fill in msg
            msg_path.ids.append(path_id)
            msg_path.center_lines.append(cl)
    if visualize_paths:
        plt.show()
    # produce pathId like col laneId 
    pathId = np.array([map_car_path[car_id] for car_id in carId])
    return msg_path, pathId

# [not used] use the graph-theoretic dual representation of the traffic topology via adjacency matrix
def compute_lc_with_paths(carId, carCenterXft, carCenterYft, laneId, 
                  npy_fn, scale_pixal_to_meters,
                  visualize_paths = True):
    veh_ids_all = np.unique(carId)
    # lane_ids = np.unique(laneId)
    lane_ids_old = np.arange(max(laneId)+1)
    _, map_car_lanes =compute_connectivity_representation(carId,laneId)
    
    # listed all the cases 
    cases = list(set(np.array(list(map_car_lanes.values()))))
    cases = sorted(cases, key=lambda x: ( len(x), x[0],x[len(x)-1])) # starting lane, ending lane, crossing lane
    
    msg_cl = CenterLanes()
    # path_ids = range(len(cases))
     
    # build a map from car_id to path_id
    map_car_path = {}
    
    # lane_ids = range(all_lane.shape[0])
    for l in lane_ids_old:
        path_ids_with_l = find_sets_containing_element(cases, l)
        path_trajs_in_l = []
        # select path taht contains the lane
        for path_id in path_ids_with_l:
            case = cases[path_id]
            cars = [car_id for car_id, lanes in map_car_lanes.items() if lanes == case]
            traj_list = []
            for car_id in cars: 
                # print(len(carCenterXft[carId == carId[i]]) >= MIN_APPEAR_TIME)
                # if len(carCenterXft[carId == carId[i]]) >= MIN_APPEAR_TIME:
                if len(carCenterXft[carId == carId[i]]) <= MIN_APPEAR_TIME:
                    print("car_id = ", car_id)
            
                map_car_path[car_id] = path_id
                mask1 = carId == id 
                mask2 = laneId == l
                xs = carCenterXft [mask1 & mask2]* FT_TO_METER
                ys = carCenterYft [mask1 & mask2]* FT_TO_METER
                ds = np.cumsum(np.sqrt(np.diff(xs,prepend=xs[0])**2 + np.diff(ys,prepend=xs[0])**2))
                xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ds - ds[0])/(ds[-1] - ds[0]), xs), 
                                np.interp(np.linspace(0,1,100), (ds - ds[0])/(ds[-1] - ds[0]), ys))
                traj_list.append([xs_int, ys_int]) 
            traj_list = np.array(traj_list)
            
            if traj_list.shape[0] != 0:
                xs_avg, ys_avg = np.mean(traj_list[:,0,:], axis=0), np.mean(traj_list[:,1,:], axis=0)
                xs_sm, ys_sm = savgol_filter(xs_avg, 21, 3),savgol_filter(ys_avg, 21, 3)
                path_trajs_in_l.append((xs_sm, ys_sm))
        
        xs_avg, ys_avg, hand_assigned_pts_is_used = assign_points_interface(path_trajs_in_l, npy_fn, l, scale_pixal_to_meters)
        
 
        if hand_assigned_pts_is_used:
            cl = PathWithSpeed()
            for i in range(xs_avg.shape[0]):
                pt = PoseStamped()
                pt.pose.position.x, pt.pose.position.y = xs_avg[i], ys_avg[i]
                cl.path.poses.append(pt)
            # fill in msg
            msg_cl.ids.append(l)
            msg_cl.center_lines.append(cl)
            
        else:
            for k, xys in enumerate(path_trajs_in_l):
                xs, ys = xys[0], xys[1]
                cl = PathWithSpeed()
                for i in range(xs_avg.shape[0]):
                    pt = PoseStamped()
                    pt.pose.position.x, pt.pose.position.y = xs[i], ys[i]
                    cl.path.poses.append(pt)
    
                # fill in msg
                msg_cl.ids.append(l*MAX_NUM_LANES + k)
                
                msg_cl.center_lines.append(cl)
    
    return msg_cl


#  Extract_center_lane from averaged trajectories
def compute_lane_centers(Ts, laneId, carId, carCenterX, carCenterY):
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


# used to debug
def visualize_problem(Ts, laneId, carId, carCenterX, carCenterY):
    lane_ids = [id for id in np.unique(laneId)]
    all_lane = np.load("dataset/McCulloch@SeminoleLanes.npy", allow_pickle=True)
    
        
    for l in lane_ids: 
        traj_list = []
        veh_ids = [id for id in np.unique(carId[laneId == l])]
        xys = all_lane[l].reshape((all_lane[l].shape[0], 2)) * 0.128070 * FT_TO_METER
        
        plt.plot(xys[:,0], xys[:,1], c = 'grey',
                alpha = 0.3)
        plt.text(xys[0,0], xys[0,1], "lc_"+str(l)) 
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
        # plt.axis("equal")
        plt.show()
        # xs_std, ys_std = np.std(traj_list[:,0,:], axis=0), np.std(traj_list[:,1,:], axis=0)
        # ds_std = np.sqrt(xs_std * xs_std + ys_std * ys_std)
        
def check_rel_difference(vx,vy, speed):
    print("calculated speed = ", np.sqrt(vx**2 + vy ** 2), "speed from raw data (m/s)", speed)

def write_msg_lane_perc(t,Ts,indices, laneId, pathId, carId, 
                        carCenterX, carCenterY, 
                        boundingBox1X, boundingBox1Y, boundingBox2X, boundingBox2Y, boundingBox3X, boundingBox3Y, 
                        heading, angle_offset):
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
                veh.local_id = pathId[i] # here we use local_id as the path id, so that we could switch to path id
                p1, p2, p3 = (boundingBox1X[i], boundingBox1Y[i]), (boundingBox2X[i], boundingBox2Y[i]), (boundingBox3X[i], boundingBox3Y[i])
                veh.length, veh.width = extract_length_width(p1, p2, p3)
                
                veh.pose.pose.orientation.x, veh.pose.pose.orientation.y, veh.pose.pose.orientation.z, veh.pose.pose.orientation.w = tf.transformations.quaternion_from_euler(0.0, 0.0, heading[i]/180 * np.pi + angle_offset) 
                # validate the trajectory is a true trajectory, that the appearance is longer than 1 sec
                if len(carCenterX[carId == carId[i]]) >= MIN_APPEAR_TIME:
                    xs, ys, ts = carCenterX[carId == carId[i]], carCenterY[carId == carId[i]], Ts[carId == carId[i]]
                    idx = np.where(ts == t)[0][0]
                    xs_sm, ys_sm = savgol_filter(xs, min(21,xs.shape[0]), 3), savgol_filter(ys, min(21,xs.shape[0]), 3)
                    # we have to calculate an estimated velocity by hand
                    veh.pose.pose.position.x, veh.pose.pose.position.y = xs_sm[idx], ys_sm[idx]
                    vx, vy = calculate_velocity(xs_sm, ys_sm, ts,idx, 8, 8)
                    # check_rel_difference(vx,vy,speed[i])
                    veh.twist.twist.linear.x, veh.twist.twist.linear.y = vx, vy
                    lane_vehs.vehicles.append(veh)
            
        msg.vehicles.append(lane_vehs)
    return  msg

def build_traffic_bag_from_data(data_csv_fn, npy_fn, scale_pixal_to_meters, output_bag_fn, start_time = 0.0, duration = 50):
    
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
        # heading calibration:
        angle_offset = calibrate_heading(carCenterXft[carId==0], carCenterYft[carId==0], heading[carId==0] *DEG_TO_RAD)
        # visualize_problem(ts, laneId, carId, carCenterXft* FT_TO_METER, carCenterYft* FT_TO_METER)
        msg_cl = hand_assign_lane_center_interface (npy_fn, scale_pixal_to_meters) #,load = False)
        
        # msg_cl = compute_lane_centers(ts, laneId, carId, carCenterXft * FT_TO_METER, carCenterYft * FT_TO_METER)
        msg_path, pathId = compute_paths(carId, carCenterXft, carCenterYft, laneId) # use the smoothed path to replace the lane center
        
        
        # print("# frames = ", np.unique(ts))
        for t in np.unique(ts):
            if t >= start_time:
                # print(ts[ts == t].shape[0])
                indices = np.where(ts == t)[0]
                # make sure all input are in meters and seconds
                
                msg_lp = write_msg_lane_perc(t, ts, indices, 
                                            laneId, #  laneId, # use the smoothed path to replace the lane center
                                            pathId,
                                            carId, 
                                    carCenterXft * FT_TO_METER, carCenterYft * FT_TO_METER, 
                                    boundingBox1Xft * FT_TO_METER, boundingBox1Yft * FT_TO_METER, boundingBox2Xft * FT_TO_METER, boundingBox2Yft * FT_TO_METER, boundingBox3Xft * FT_TO_METER, boundingBox3Yft * FT_TO_METER, 
                                    heading, angle_offset)


                # bag.write('/region/lanes_center', msg_cl_filtered, rospy.Time(t))
                
                # bag.write('/region/traffic_avg_paths', msg_lp, rospy.Time(t))
                bag.write('/region/lanes_center', msg_cl, rospy.Time(t))
                bag.write('/region/traffic_path', msg_path, rospy.Time(t))
                bag.write('/region/lanes_perception', msg_lp, rospy.Time(t))
                # this topic includes avg path 
                
                print("msg sent at t = ", t)
                if t >= (start_time + duration):
                    break      
    finally:
        bag.close()
        
    

def calibrate_heading(xs, ys, hds):
    dxs, dys = np.diff(xs, append=xs[-1]), np.diff(ys,append=ys[-1])
    vel_dir = np.arctan2(dys, dxs)
    angle_diff = angle_wrap_0_2pi_from_np(np.mean(vel_dir - hds))
    return np.pi/2 * (round(angle_diff/(np.pi/2)))%4

def angle_wrap_0_2pi_from_np(angle_radian):
    if angle_radian < 0: 
        return angle_wrap_0_2pi_from_np(angle_radian + np.pi * 2)
    elif angle_radian >= 2 * np.pi:
        return angle_wrap_0_2pi_from_np(angle_radian - np.pi * 2)
    else:
        return angle_radian    

if __name__ == "__main__":
    # graph representation of lanes
    input_csv_fn = "dataset/McCulloch@Seminole-01.csv"
    input_npy_fn = "dataset/McCulloch@SeminoleLanes.npy"
    # input_csv_fn = 'dataset/RoundaboutA-02.csv'
    # input_npy_fn = "dataset/county@oviedoLanes.npy" #"dataset/RoundaboutALane.npy"
    # input_csv_fn = "dataset/ExpresswayA-01.csv"
    # input_npy_fn = "dataset/ExpresswayALanes.npy"
    
    # msg_path = compute_paths(input_csv_fn)
    # cases to pass the regions
    # output_bag_fn = "bags/RoundaboutA_gt_use_path.bag"
    output_bag_fn = "bags/McCulloch_gt_use_both.bag"
    s = 0.128070 * FT_TO_METER
    build_traffic_bag_from_data(input_csv_fn, input_npy_fn,s, output_bag_fn, duration = 50)
        
    