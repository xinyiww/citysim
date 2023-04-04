
import numpy as np

import matplotlib.pyplot as plt




# get t, veh_id, pos.x, pos.y
def getGTData(filename):
    file = open(filename, 'r')
    dat_str = file.readlines()
    data = np.zeros([len(dat_str), 4])
    for i in range(len(dat_str)):
        sample = dat_str[i].split(' ')
#         print(sample)
        if len(sample) != 4:
            print ('!check at line ', str(i), ' of file ', str(filename))
        else:
            
            t, veh_id, x, y= (float(sample[i]) for i in range(4))
            
            data[i, :] = t, veh_id, x, y
    return data

# get t_start,dt,n_pred,i | xs,ys,dxs, dys
def getPredData(filename):
    file = open(filename, 'r')
    dat_str = file.readlines()
    import matplotlib.pyplot as plt
    sample = dat_str[0].split(' ')
    n_pred= int(sample[2])
    width = 4+4*n_pred
    # print(width)       
    data = np.zeros([len(dat_str), width]) 
    for i in range(len(dat_str)):
        sample = dat_str[i].split(' ')
        data[i, :] = sample
            
    return data

if __name__ == "__main__":
    getLaneRaw("/home/xliu/Documents/dataset/CitySim/dataset/McCulloch@SeminoleLanes.npy")

