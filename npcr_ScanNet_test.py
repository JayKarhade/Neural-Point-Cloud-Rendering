from numpy.core.fromnumeric import shape
from network_pytorch import *
import cv2, os, time, math
import glob
import scipy.io as io
from loss_pytorch import *
from utils import *

##Pytorch Imports
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

is_training = False  # if test, set this 'False'
use_viewdirection = True  # use view direction
renew_input = False   # optimize input point features.
constant_initial = True  # use constant value for initialization.
use_RGB = True     # use RGB information for initialization.
random_crop = True  # crop image.

d = 32   # how many planes are used, identity with pre-processing.
h = 480  # image height, identity with pre-processing.
w = 640  # image width, identity with pre-processing.
top_left_v = 0#120  # top left position
top_left_u = 0#160  # top left position
h_croped = 480#240  # crop size height
w_croped = 640#320  # crop size width
forward_time = 4  # optimize input point features after cropping 4 times on one image.
overlap = 32  # size of overlap region of crops.

channels_i = int(8)  # dimension of input point features
channels_o = 3  # output image dimensions
channels_v = 3  # view direction dimensions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_id = 0
num_epoch = 21
decrease_epoch = 7  # epochs, learning_rate_1 decreased.
learning_rate = 0.0001  # learning rate for network parameters optimization
learning_rate_1 = 0.01  # initial learning rate for input point features.

dataset = 'ScanNet'     # datasets
scene = 'scene0010_00'  # scene name
root = '/content/drive/MyDrive/Neural-Point-Cloud-Rendering/'
task = root+'%s_npcr_%s' % (dataset, scene)  # task name, also path of checkpoints file
dir1 = root+'data/%s/%s/color/' % (dataset, scene)  # path of color image
dir2 = root+'data/%s/%s/pose/' % (dataset, scene)  # path of camera poses.
dir3 = root+'pre_processing_results/%s/%s/reproject_results_%s/' % (dataset, scene, d)  # voxelization information path.
dir4 = root+'pre_processing_results/%s/%s/weight_%s/' % (dataset, scene, d)  # aggregation information path.
dir5 = root+'pre_processing_results/%s/%s/point_clouds_simplified.ply' % (dataset, scene)  # point clouds file path

PATH = '/content/drive/MyDrive/Neural-Point-Cloud-Rendering/ScanNet_npcr_scene0010_00/model_pytorch'
state = torch.load(PATH)
model = UNet()
model.to(device)
model.load_state_dict(state['model_state_dict'])

num_image = len(glob.glob(os.path.join(dir1, '*.jpg')))

image_names_train, index_names_train, camera_names_train, index_names_1_train,\
image_names_test, index_names_test, camera_names_test, index_names_1_test = prepare_data_ScanNet(dir1, dir2, dir3, dir4, num_image)

# load point clouds information
point_clouds, point_clouds_colors = loadfile(dir5)
num_points = point_clouds.shape[1]

# initial descriptor
descriptors = np.random.normal(0, 1, (1, num_points, channels_i))

if os.path.isfile('%s/descriptorpytorch.mat' % task):
    content = io.loadmat('%s/descriptorpytorch.mat' % task)
    descriptors = content['descriptors']
    print('loaded descriptors.')
else:
    if constant_initial:
        descriptors = np.ones((1, num_points, channels_i), dtype=np.float32) * 0.5

    if use_RGB:
        descriptors[0, :, 0:3] = np.transpose(point_clouds_colors) / 255.0

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_id

#input1 = torch.tensor([1,channels_i,d,-1,-1],dtype=torch.float32)
#input2 = torch.tensor([1,channels_v,d,-1,-1],dtype=torch.float32)
#output = torch.tensor([1,channels_i,d,-1,-1],dtype=torch.float32)


if not is_training:

    output_path = "%s/TestResultpytorch/" % (task)
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

        # load weight
        npzfile_weight = np.load(index_name_1)
        weight = npzfile_weight['weight_average']
        distance_to_depth_min = npzfile_weight['distance_to_depth_min']

        extrinsic_matrix = CameraPoseRead(camera_name)  # camera to world
        camera_position = np.transpose(extrinsic_matrix[0:3, 3])

        max_num = np.max(each_split_max_num)
        group_descriptor = np.zeros([(max(group_belongs + 1)), max_num, channels_i], dtype=np.float32)
        group_descriptor[group_belongs, index_in_each_group, :] = descriptors[0, select_index, :] * np.expand_dims(weight, axis=1)

        image_descriptor[0, n, v, u, :] = np.sum(group_descriptor, axis=1)[group_belongs, :]

        view_direction[0, n, v, u, :] = np.transpose(point_clouds[0:3, select_index]) - camera_position
        view_direction[0, n, v, u, :] = view_direction[0, n, v, u, :] / (
        np.tile(np.linalg.norm(view_direction[0, n, v, u, :], axis=1, keepdims=True), (1, 3)) + 1e-10)


        ##Evalute result via network code to enter here
        #.[result] = sess.run([network], feed_dict={input1: image_descriptor, input2: view_direction})
        #.result = np.minimum(np.maximum(result, 0.0), 1.0) * 255.0
        #Assumes PATH provided
        #Concatenate image descriptor and view_direction inputs
        image_descriptor = torch.from_numpy(image_descriptor).permute(0,4,1,2,3).to(device)
        view_direction = torch.from_numpy(view_direction).permute(0,4,1,2,3).to(device)

        data = torch.cat((image_descriptor[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)],
                          view_direction[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)]),1).to(device)
        model.eval()
        result = model(data.float())
        result = result[2]
        result = result.permute(0,2,3,1)
        result = result.detach().to('cpu').numpy()#.squeeze().data.cpu().numpy()
        #.
        ####
        result = np.minimum(np.maximum(result, 0.0), 1.0) * 255.0
        print(result[0,:,:,:].shape)
        cv2.imwrite(output_path + '%06d.png' % id, np.uint8(result[0, :, :, :]))

if __name__=='__main__':
    pass
