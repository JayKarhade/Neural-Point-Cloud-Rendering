from network import *
import cv2, os, time, math
import glob
import scipy.io as io
from loss import *
from utils import *

##Pytorch Imports
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

is_training = False  # if test, set this 'False'
use_viewdirection = True  # use view direction
renew_input = True   # optimize input point features.
constant_initial = True  # use constant value for initialization.
use_RGB = True     # use RGB information for initialization.
random_crop = True  # crop image.

d = 32   # how many planes are used, identity with pre-processing.
h = 480  # image height, identity with pre-processing.
w = 640  # image width, identity with pre-processing.
top_left_v = 0  # top left position
top_left_u = 0  # top left position
h_croped = 240  # crop size height
w_croped = 320  # crop size width
forward_time = 4  # optimize input point features after cropping 4 times on one image.
overlap = 32  # size of overlap region of crops.

channels_i = int(8)  # dimension of input point features
channels_o = 3  # output image dimensions
channels_v = 3  # view direction dimensions

gpu_id = 3
num_epoch = 21
decrease_epoch = 7  # epochs, learning_rate_1 decreased.
learning_rate = 0.0001  # learning rate for network parameters optimization
learning_rate_1 = 0.01  # initial learning rate for input point features.

dataset = 'ScanNet'     # datasets
scene = 'scene0010_00'  # scene name
task = '%s_npcr_%s' % (dataset, scene)  # task name, also path of checkpoints file
dir1 = 'data/%s/%s/color/' % (dataset, scene)  # path of color image
dir2 = 'data/%s/%s/pose/' % (dataset, scene)  # path of camera poses.
dir3 = 'pre_processing_results/%s/%s/reproject_results_%s/' % (dataset, scene, d)  # voxelization information path.
dir4 = 'pre_processing_results/%s/%s/weight_%s/' % (dataset, scene, d)  # aggregation information path.
dir5 = 'pre_processing_results/%s/%s/point_clouds_simplified.ply' % (dataset, scene)  # point clouds file path

num_image = len(glob.glob(os.path.join(dir1, '*.jpg')))

image_names_train, index_names_train, camera_names_train, index_names_1_train,\
image_names_test, index_names_test, camera_names_test, index_names_1_test = prepare_data_ScanNet(dir1, dir2, dir3, dir4, num_image)

# load point clouds information
point_clouds, point_clouds_colors = loadfile(dir5)
num_points = point_clouds.shape[1]

# initial descriptor
descriptors = np.random.normal(0, 1, (1, num_points, channels_i))

if os.path.isfile('%s/descriptor.mat' % task):
    content = io.loadmat('%s/descriptor.mat' % task)
    descriptors = content['descriptors']
    print('loaded descriptors.')
else:
    if constant_initial:
        descriptors = np.ones((1, num_points, channels_i), dtype=np.float32) * 0.5

    if use_RGB:
        descriptors[0, :, 0:3] = np.transpose(point_clouds_colors) / 255.0

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_id


def test_epoch(net,loader):
    output_path = "%s/TestResult/" % (task)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    for id in range(len(camera_names_test)):

        st = time.time()
        image_descriptor = np.zeros([1,d,h,w,channels_i])
        view_direction = np.zeros([1,d,h,w,channels_v])
        camera_name = camera_names_test[id]
        index_name = index_names_test[id]
        index_name_1 = index_names_1_test[id]

        if not (os.path.isfile(index_name) and os.path.isfile(camera_name) and os.path.isfile(index_name_1)):
            print('Missingg file 1!')
            continue

        npzfile = np.load(index_name)
        u = npzfile['u']
        v = npzfile['v']
        n = npzfile['d']
        select_index = npzfile['select_index']
        group_belongs = npzfile['group_belongs']
        index_in_each_group = npzfile['index_in_each_group']
        distance = npzfile['distance']
        each_split_max_num = npzfile['each_split_max_num']

        max_num = np.max(each_split_max_num)
        group_descriptor = np.zeros([(max(group_belongs + 1)), max_num, channels_i], dtype=np.float32)
        group_descriptor[group_belongs, index_in_each_group, :] = descriptors[0, select_index, :] * np.expand_dims(weight, axis=1)

        image_descriptor[0, n, v, u, :] = np.sum(group_descriptor, axis=1)[group_belongs, :]

        view_direction[0, n, v, u, :] = np.transpose(point_clouds[0:3, select_index]) - camera_position
        view_direction[0, n, v, u, :] = view_direction[0, n, v, u, :] / (
        np.tile(np.linalg.norm(view_direction[0, n, v, u, :], axis=1, keepdims=True), (1, 3)) + 1e-10)

        ##Evalute result via network code to enter here
        #.
        #.
        #.
        ####

        result = np.minimum(np.maximum(result, 0.0), 1.0) * 255.0
        cv2.imwrite(output_path + '%06d.png' % id, np.uint8(result[0, :, :, :]))

if __name__=='__main__':
    pass