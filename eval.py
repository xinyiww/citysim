import sys
import os
import csv
import math
# import rosbag
import rospy
import copy
import numpy as np
from dataImport import *
import matplotlib.pyplot as plt

def run_ADE_eval(folder_path, PLOT_ERROR_DIST=True, PLOT_ONE_TRAJECTORY = True, RUN_CALCULATION =  True, ratio_hrz=1): 
    GT = []
    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename[:2] == 'gt':
            GT = getGTData(file_path)
            veh_ids = np.sort(np.unique(GT[:,1])).astype(int)
    
    if GT == []:
        print("No GT data found. End eval.")
    
    if PLOT_ERROR_DIST and RUN_CALCULATION:
        fig_error= plt.figure(1)
        ax = fig_error.add_subplot(111)
        ax.set_xlabel('Sample Index (sorted by error)')
        ax.set_ylabel('Error')
        ax.set_title('Sorted prediction error distribution')
    
    if PLOT_ONE_TRAJECTORY:   
        plot_count = 0
        filename_list = ['ct_vel', 'GM', 'idm']
        fig_xy_vs_t = plt.figure()
        ax1 = fig_xy_vs_t.add_subplot(311) 
        ax2 = fig_xy_vs_t.add_subplot(312)
        ax3 = fig_xy_vs_t.add_subplot(313)
        axs = [ax1, ax2, ax3]
        ax3.set_xlabel ("t")
        
        
          
    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        
        if filename[:4] == 'pred':
            
            PRED = getPredData(file_path) 
            dt, n_pred = PRED[0,1:3]
            n_pred = int(n_pred)
            hrz = dt * n_pred
            
            # if filename == "pred_GM.csv":
            #     PRED[:,0] += dt

            running_l2e_all = 0
            counts_all = 0
            preds_pw_mean_std = [] 
            pred_all_vs_t = []
            
            preds_all = []
            
            # for i, (veh_id, veh_id) in enumerate(id_gt2pred.items()):
            for veh_id in veh_ids:
                
                running_l2e_traj = 0
                counts_traj = 0
                data_gt = GT[GT[:,1] == float(veh_id)]
                txys_gt = data_gt[:, (0,2,3)]
                data_pred = PRED[PRED[:,3] == float(veh_id)] # t_start,dt,n_pred,i | xs,ys,dxs, dys
                
                ### we might want to visualize the trajectories a bit 
                if PLOT_ONE_TRAJECTORY and (veh_id == veh_ids[2]) and filename[5:-4] in filename_list:
                    
                    axs[plot_count].set_ylim([-180,0])
                    axs[plot_count].plot(txys_gt[:,0], txys_gt[:,2], '-o',markersize=7, label = 'GT')
                    asp = np.diff(axs[0].get_xlim())[0] / np.diff(axs[0].get_ylim())[0]
                    axs[plot_count].set_aspect("auto")
                    axs[plot_count].set_title(filename[5:-4])
                    axs[plot_count].set_ylabel("longitute (m)")
                    axs[plot_count].legend()
                    for j in range(data_pred.shape[0]):
                        if j % 10 == 0:
                            t_start = data_pred[j,0]
                            ts = np.linspace(t_start, t_start + hrz, n_pred)
                            xs = data_pred[j, 4+ n_pred: 4+ n_pred * 2]
                            sigs = data_pred[j, 4+ +n_pred * 2: 4+n_pred * 3]
                            axs[plot_count].scatter(ts, xs, s = 4, marker = '*')
                            axs[plot_count].fill_between(ts, xs - sigs , xs + sigs, color='gray', alpha=0.1)
                    plot_count += 1
                
                if RUN_CALCULATION:
                    for txy_gt in txys_gt:
                        running_l2e_pt = 0
                        t_gt = txy_gt[0]
                        pred = data_pred [ data_pred[:,0] <= t_gt]
                        pred = pred [pred[:,0] >= t_gt - hrz*ratio_hrz ] 
                        l2es_temp = []
                        if pred.shape[0] > 0 :
                            for pred_traj in pred:
                                t_start = pred_traj[0] 
                                xs = pred_traj[4: 4+n_pred]
                                ys = pred_traj[4+n_pred: 4+(n_pred)*2]
                                x_pred = np.interp(t_gt, np.linspace(t_start, t_start + hrz, n_pred), xs)
                                y_pred = np.interp(t_gt, np.linspace(t_start, t_start + hrz, n_pred), ys)
                                l2e = np.sqrt((txy_gt[1] - x_pred) **2 +  (txy_gt[2] - y_pred) **2)
                                l2es_temp.append(l2e)
                                preds_all.append(l2e)
                                running_l2e_pt += l2e
                                pred_all_vs_t.append([t_gt - pred_traj[0], l2e])
                            
                            preds_pw_mean_std.append([np.mean(np.array(l2es_temp)), np.std(np.array(l2es_temp))])
                            # print("avg point-wise ade = ", running_l2e_pt/ pred.shape[0])
                        
                        running_l2e_traj += running_l2e_pt
                        counts_traj+= pred.shape[0]
                if RUN_CALCULATION: 
                    # print("avg trajectory-wise ade = ", running_l2e_traj/ counts_traj)
                    running_l2e_all += running_l2e_traj
                    counts_all += counts_traj
            if RUN_CALCULATION:     
                print("avg ADE of "+ file_path +" = ", running_l2e_all/ counts_all)    

        
            if PLOT_ERROR_DIST and RUN_CALCULATION:
                preds_pw_mean_std = np.array(preds_pw_mean_std)
                preds_pw_mean_std_sorted = preds_pw_mean_std[preds_pw_mean_std[:, 0].argsort()] 
                # plt.errorbar(x, preds_pw_mean_std_sorted[:,0], yerr=preds_pw_mean_std_sorted[:,1], fmt='-', capsize=5, markersize=5)
                x = np.linspace(0,1,preds_pw_mean_std_sorted.shape[0])
                # plt.fill_between(x, preds_pw_mean_std_sorted[:,0] - preds_pw_mean_std_sorted[:,1] , preds_pw_mean_std_sorted[:,0] + preds_pw_mean_std_sorted[:,1], color='gray', alpha=0.5)
                # ax.plot(x, preds_pw_mean_std_sorted[:,0], '-o', markersize=3,label = filename[:-4])
                preds_all = np.array(preds_all)
                print(preds_pw_mean_std_sorted.shape, preds_all.shape)
                
                preds_all_sorted = preds_all[preds_all.argsort()]
                x = np.linspace(0,1,preds_all_sorted.shape[0])
                ax.plot(x, preds_all_sorted, '-o', markersize=3,label = filename[:-4]+", ADE ="+f"{running_l2e_all/ counts_all:.2f}")
                ax.set_xlabel("sorted ADE index percentile (horizon = "+f"{hrz*ratio_hrz:.2f}"+" sec, #dp/all = "+str(counts_all)+"/"+str(GT.shape[0]*n_pred)+")")
                ax.legend()
                
            
    plt.show()   
    
    
    # ### FDE: 
    
    
    # The final displacement error (FDE) is the L2 distance between the final points of the 
    # prediction and ground truth. We take the minimum FDE over the k most likely predictions 
    # and average over all agents.

    # # - For each selected GT trajectory's end point (veh_id):

    # #     - Count how many prediction trajectories' time range covers the time stamp of this GT point 
    # #     - For each prediction trajectory:
    # #         - Calulate a linear interpolation on that GT point
    # #         - store it in order of predicted length in a list, which direct to the GT point
            
def run_FDE_eval(folder_path, PLOT_ERROR_DIST, ratio_hrz=1): 
    GT = []
    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename[:2] == 'gt':
            GT = getGTData(file_path)
            veh_ids = np.sort(np.unique(GT[:,1])).astype(int)
    
    if GT == []:
        print("No GT data found. End eval.")
        
    if PLOT_ERROR_DIST:
            fig_error= plt.figure(1)
            ax = fig_error.add_subplot(111)
            ax.set_ylabel('Error')
            ax.set_title('Sorted prediction error distribution')
            
                
        
    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
    
        if filename[:4] == 'pred':
            
            PRED = getPredData(file_path) 
            dt, n_pred = PRED[0,1:3]
            n_pred = int(n_pred)
            hrz = dt * n_pred
            

            running_l2e_all = 0
            counts_all = 0
            err = []
            
            for veh_id in veh_ids:
                
                data_gt = GT[GT[:,1] == float(veh_id)]
                txys_gt = data_gt[:, (0,2,3)]
                data_pred = PRED[PRED[:,3] == float(veh_id)] # t_start,dt,n_pred,i | xs,ys,dxs, dys
                counts_veh = 0
                # txy_pt = txys_gt[txys_gt.shape[0]-1]
                running_l2e_pt = 0
                for pred_traj in data_pred:
                    # linear interp
                    t_start = pred_traj[0]
                    ind = math.floor(n_pred* ratio_hrz)
                    # print("ratio= ", ratio_hrz, "index = ", ind)
                    xf = pred_traj[4 + ind -1]
                    yf = pred_traj[4 + n_pred +ind -1]
                    if t_start + hrz * ratio_hrz <= txys_gt[-1,0]:
                        xf_gt_int = np.interp(t_start + hrz * ratio_hrz, txys_gt[:,0], txys_gt[:,1])
                        yf_gt_int = np.interp(t_start + hrz * ratio_hrz, txys_gt[:,0], txys_gt[:,1])
                        l2e_pt =  np.sqrt((xf_gt_int - xf) **2 +  (yf_gt_int - yf) **2)
                        running_l2e_pt += l2e_pt
                        counts_veh += 1
                        err.append(l2e_pt)
                running_l2e_all += running_l2e_pt
                counts_all += counts_veh
                
            print("FDE " + file_path +" =  ",running_l2e_all/ counts_all)
            if PLOT_ERROR_DIST:
                err = np.array(err)
                err_sorted= err[err.argsort()]
                # plt.errorbar(x, preds_pw_mean_std_sorted[:,0], yerr=preds_pw_mean_std_sorted[:,1], fmt='-', capsize=5, markersize=5)
                x = np.linspace(0,1,err.shape[0])

                ax.plot(x, err_sorted, '-o', markersize=3,label = filename[:-4]+", FDE ="+f"{running_l2e_all/ counts_all:.2f}")
                ax.legend()
                ax.set_xlabel("sorted FDE index percentile (horizon = "+f"{hrz*ratio_hrz:.2f}"+" sec, #dp/all = "+str(counts_all)+"/"+str(PRED.shape[0])+")")
                
    plt.show()
                       



if __name__ == '__main__':
    # result_folder = sys.argv[1]
    result_folder = "bags/McCulloch_pred_results"
    # run_ADE_eval(result_folder, RUN_CALCULATION =  True, PLOT_ONE_TRAJECTORY=False, PLOT_ERROR_DIST=True, ratio_hrz=0.5) 
    run_FDE_eval(result_folder, PLOT_ERROR_DIST=True, ratio_hrz=0.5)
    