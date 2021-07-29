import numpy as np
import os
import random
from scipy.spatial.transform import Rotation as R
from pre_processing.voxelization_aggregation_ScanNet import *

##This file currently only samples poses. Direct input generation will be added very soon.


num_aug_poses = int(input("Enter Number of augmented poses you want from each image: "))
posedir = '/content/drive/MyDrive/Neural-Point-Cloud-Rendering/data/ScanNet/scene0010_00/pose/'

pose_index = np.random.randint(low=1,high=100,size=5)
print(pose_index)
cam_mat_list = []

for i in range(len(pose_index)):
    idx = pose_index[i]
    x = np.genfromtxt(posedir+str(idx)+'.txt')
    for j in range(num_aug_poses):
        ##x is the pose 
        cam_matrix_new = x.copy()##array to store new poses

        rotmat = x[:3,:3].copy()#we perturb rotations and not translations
        #add random angles to euler vector for new pose
        r  = R.from_matrix(rotmat)
        new_euler = r.as_euler('zyx',degrees=True)
        new_euler = new_euler + (np.random.rand(1,3)*360)
        r_new = R.from_euler('zyx',new_euler)
        ##new perturbed pose
        cam_matrix_new[:3,:3] = r_new.as_matrix()
        cam_mat_list.append(x)

cam_mat_final = np.asarray(cam_mat_list)
#print(cam_mat_final)
np.save("/content/drive/MyDrive/Neural-Point-Cloud-Rendering/generated_poses/aug_poses.npy",cam_mat_final)

if not os.path.isdir('/content/drive/MyDrive/Neural-Point-Cloud-Rendering/generated_poses'):
    os.makedirs('/content/drive/MyDrive/Neural-Point-Cloud-Rendering/generated_poses')

for i in range(len(cam_mat_list)):
    filename = '/content/drive/MyDrive/Neural-Point-Cloud-Rendering/'+'generated_poses/'+str(i)+'.txt'
    np.savetxt(filename,cam_mat_list[i])

