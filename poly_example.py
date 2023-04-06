from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

from bag_from_citysim import *
def compute_lane_centers_polygon(npy_fn):
    # # Create an image, finding l, w
    # r = 100
    # src = np.zeros((4*r, 4*r), dtype=np.uint8)
    # # Create a sequence of points to make a contour
    # vert = [None]*6
    # vert[0] = (3*r//2, int(1.34*r))
    # vert[1] = (1*r, 2*r)
    # vert[2] = (3*r//2, int(2.866*r))
    # vert[3] = (5*r//2, int(2.866*r))
    # vert[4] = (3*r, 2*r)
    # vert[5] = (5*r//2, int(1.34*r))
    
    
    
    
    all_lane = np.load(npy_fn, allow_pickle=True)
    
    
    for lane_id in range(all_lane.shape[0]):
        xys = all_lane[lane_id].reshape((all_lane[lane_id].shape[0], 2))#* 0.128070 * 0.3048
        # xys = np.append(xys, np.array([[xys[0,0], xys[0,1]]]), axis = 0)
        padding = 5
        xlb, ylb = np.max(xys[:,0])- np.min(xys[:,0]) + padding * 2, np.max(xys[:,1])- np.min(xys[:,1]) + padding * 2# local bound
        src = np.zeros((ylb, xlb), dtype=np.uint8)
        xls, yls = xys[:,0] - np.min(xys[:,0])+ padding, xys[:,1] - np.min(xys[:,1])  + padding
        # plt.scatter(xls, yls)
        # plt.plot(xls, yls)
        vert = [(xls[i],yls[i]) for i in range(xls.shape[0]) ]
        # # plt.plot(xys[:,0], xys[:,1], c = 'grey', alpha = 0.3)
        # # plt.show()
        # # Draw it in src
        for i in range(xls.shape[0]):
            cv.line(src, vert[i],  vert[(i+1)% xls.shape[0]], ( 255 ), 3)
        # Get the contours
        contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Calculate the distances to the contour
        raw_dist = np.empty(src.shape, dtype=np.float32)
        
        
        # for i in range(src.shape[0]):
        #     for j in range(src.shape[1]):
        #         raw_dist[i,j] = cv.pointPolygonTest(contours[0], (j,i), True)
        # minVal, maxVal, _, maxDistPt = cv.minMaxLoc(raw_dist)
        # minVal = abs(minVal)
        # maxVal = abs(maxVal)
        # # Depicting the distances graphically
        # drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        # for i in range(src.shape[0]):
        #     for j in range(src.shape[1]):
        #         if raw_dist[i,j] > 0 and raw_dist[i,j] > maxVal - 30:
        #             drawing[i,j,2] = 255 - raw_dist[i,j] * 255 / maxVal # red, positive
        #         elif raw_dist[i,j] < 0:
        #             drawing[i,j,0] = 255 - abs(raw_dist[i,j]) * 255 / minVal
        #         else:
        #             drawing[i,j,0] = 255 
        #             drawing[i,j,1] = 255
        #             drawing[i,j,2] = 255
        # # cv.circle(drawing,maxDistPt, int(maxVal),(255,255,255), 1, cv.LINE_8, 0)
        # # cv.imshow('Source', src)
        # plt.imshow(cv.cvtColor(drawing,  cv.COLOR_BGR2RGB))
        plt.show()
        print("lane "+str(lane_id))
        # cv.waitKey()

if __name__ == "__main__":
    # graph representation of lanes
    # input_csv_fn = "dataset/McCulloch@Seminole-01.csv"
    input_npy_fn = "dataset/McCulloch@SeminoleLanes.npy"
    input_csv_fn = 'dataset/RoundaboutA-02.csv'
    # input_npy_fn = "dataset/county@oviedoLanes.npy" #"dataset/RoundaboutALane.npy"
    # input_csv_fn = "dataset/ExpresswayA-01.csv"
    # input_npy_fn = "dataset/ExpresswayALanes.npy"
    
    # msg_path = compute_paths(input_csv_fn)
    # cases to pass the regions
    output_bag_fn = "bags/RoundaboutA_gt_use_path.bag"
   
   
    # build_traffic_bag_from_data(input_csv_fn, output_bag_fn)
        
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_fn)
    
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
    
    compute_lane_centers_polygon(input_npy_fn)
