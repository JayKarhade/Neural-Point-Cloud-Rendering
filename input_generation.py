import numpy as np
import os
import random
from scipy.spatial.transform import Rotation as R
from pre_processing.voxelization_aggregation_ScanNet import *

num_aug_poses = int(input("Enter Number of augmented poses you want from each image: "))
low_idx = int(input('Enter smallest file index for sampling: '))
high_idx = int(input('Enter highest file index for sampling'))
num_file_samples = int(input("Enter number of files you want to sample from"))
posedir = '/content/drive/MyDrive/Neural-Point-Cloud-Rendering/data/ScanNet/scene0010_00/pose/'

pose_index = np.random.randint(low=low_idx,high=high_idx,size=num_file_samples)
print(pose_index)
cam_mat_list = []

for i in range(len(pose_index)):
    idx = int(pose_index[i])
    x = np.genfromtxt(posedir+str(idx)+'.txt')
    for j in range(num_aug_poses):
        ##x is the pose 
        cam_matrix_new = x.copy()##array to store new poses

        rotmat = x[:3,:3].copy()#we perturb rotations and not translations
        #add random angles to euler vector for new pose
        r  = R.from_matrix(rotmat)
        new_euler = r.as_euler('zyx',degrees=True)
        print(new_euler)
        #print(r.as_matrix())
        new_euler_pose = new_euler + (np.random.rand(1,3)*45)
        r_new = R.from_euler('zyx',new_euler_pose,degrees=True)
        ##new perturbed pose
        print(new_euler_pose)
        #print(r_new.as_matrix())
        cam_matrix_new[:3,:3] = r_new.as_matrix()
        cam_mat_list.append(cam_matrix_new)

cam_mat_final = np.asarray(cam_mat_list)
#print(cam_mat_final)
#np.save("/content/drive/MyDrive/Neural-Point-Cloud-Rendering/generated_poses/aug_poses.npy",cam_mat_final)

if not os.path.isdir('/content/drive/MyDrive/Neural-Point-Cloud-Rendering/generated_poses'):
    os.makedirs('/content/drive/MyDrive/Neural-Point-Cloud-Rendering/generated_poses')

for i in range(len(cam_mat_list)):
    #print('original',np.genfromtxt(posedir+str(i)+'.txt'))
    #print('new',cam_mat_list[i])
    filename = '/content/drive/MyDrive/Neural-Point-Cloud-Rendering/'+'generated_poses/'+str(i)+'.txt'
    #np.savetxt(filename,cam_mat_list[i])

