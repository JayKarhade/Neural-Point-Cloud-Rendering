from numpy.core.fromnumeric import shape
from torch.optim import optimizer
from network_pytorch import *
import cv2, os, time, math
import glob
import scipy.io as io
from loss_pytorch import *
from utils import *

##Pytorch Imports
import torch,gc
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

gc.collect()
torch.cuda.empty_cache()

is_training = True  # if test, set this 'False'
use_viewdirection = True  # use view direction
renew_input = False   # optimize input point features.
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

gpu_id = 0
num_epoch = 21
decrease_epoch = 7  # epochs, learning_rate_1 decreased.
learning_rate = 0.0001  # learning rate for network parameters optimization
learning_rate_1 = 0.01  # initial learning rate for input point features.

dataset = 'ScanNet'     # datasets
scene = 'scene0010_00'  # scene name
root = '/home/zhy/Desktop/jay/PCD_Rendering/Neural-Point-Cloud-Rendering-via-Multi-Plane-Projection/'
task = root+'%s_npcr_%s' % (dataset, scene)  # task name, also path of checkpoints file
dir1 = root+'data/%s/%s/color/' % (dataset, scene)  # path of color image
dir2 = root+'data/%s/%s/pose/' % (dataset, scene)  # path of camera poses.
dir3 = root+'pre_processing_results/%s/%s/reproject_results_%s/' % (dataset, scene, d)  # voxelization information path.
dir4 = root+'pre_processing_results/%s/%s/weight_%s/' % (dataset, scene, d)  # aggregation information path.
dir5 = root+'pre_processing_results/%s/%s/point_clouds_simplified.ply' % (dataset, scene)  # point clouds file path

#tensorboard dir
writer = SummaryWriter(root+'runs/test')

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

#os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_id

model = UNet()
model.cuda()
opt = optim.Adam(model.parameters(),lr=learning_rate)

#state  = torch.load('/home/zhy/Desktop/jay/PCD_Rendering/Neural-Point-Cloud-Rendering-via-Multi-Plane-#Projection/ScanNet_npcr_scene0010_00/model_pytorch')
#if state:
#  print('found previous checkpoint')
#  model.load_state_dict(state['model_state_dict'])
#  opt.load_state_dict(state['optimizer_state_dict'])

def adjust_learning_rate(optimizer, lrd):
    for param_group in optimizer.param_groups:
        print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
        param_group['lr'] *= lrd

if is_training==True:
    model.train()
    print('begin training!')
    all = np.zeros(20000, dtype=float)
    cnt = 0

    for epoch in range(num_epoch):
        #print(epoch)
        if epoch >= decrease_epoch:
            learning_rate_1 = 0.005
            adjust_learning_rate(opt,learning_rate_1)

        if epoch >= decrease_epoch*2:
            learning_rate_1 = 0.001
            adjust_learning_rate(opt,learning_rate_1)

        if os.path.isdir("%s/%04d" % (task, epoch)):
            print("checkpoint exists")
            continue
        else:
          os.makedirs("%s/%04d" % (task, epoch))
        print("train data len",len(image_names_train))
        print("test data len",len(image_names_test))
        
        for i in np.random.permutation(len(image_names_train)):
        # for i in range(4):
            st = time.time()
            image_descriptor = np.zeros([1, d, h, w, channels_i], dtype=np.float32)
            view_direction = np.zeros([1, d, h, w, channels_v], dtype=np.float32)
            input_gradient_all = np.zeros([1, d, h, w, channels_i], dtype=np.float32)
            #input_gradient_all = torch.from_numpy(input_gradient_all).cuda()
            count = np.zeros([1, d, h, w, 1], dtype=np.float32)
            #count = torch.from_numpy(count).cuda()
            camera_name = camera_names_train[i]
            index_name = index_names_train[i]
            image_name = image_names_train[i]
            index_name_1 = index_names_1_train[i]

            if not (os.path.isfile(camera_name) and os.path.isfile(image_name) and os.path.isfile(index_name) and os.path.isfile(index_name_1)):
                print("Missing file!")
                continue

            # we pre-process the voxelization and aggregation, in order to save time.
            npzfile = np.load(index_name)
            u = npzfile['u']  # u position on image plane
            v = npzfile['v']  # v position on image plane
            n = npzfile['d']  # indicates which plane
            select_index = npzfile['select_index']   # select index of all points.
            group_belongs = npzfile['group_belongs']  # points belong to which group/voxel
            index_in_each_group = npzfile['index_in_each_group']  # index in each group/voxel
            distance = npzfile['distance']  # distance to grid center
            each_split_max_num = npzfile['each_split_max_num']  # max num of points in one group/voxel in each plane.

            # load weight
            npzfile_weight = np.load(index_name_1)
            weight = npzfile_weight['weight_average']  # normalized weights for points aggregation.
            distance_to_depth_min = npzfile_weight['distance_to_depth_min']  # distance to minimum depth value in one group/voxel.

            # calculate update weight of each point feature
            descriptor_renew_weight = (1-distance)*(1/(1+distance_to_depth_min))

            extrinsic_matrix = CameraPoseRead(camera_name)  # camera to world
            camera_position = np.transpose(extrinsic_matrix[0:3, 3])

            max_num = np.max(each_split_max_num)  # max number of points in all group/voxel
            group_descriptor = np.zeros([(max(group_belongs+1)), max_num, channels_i], dtype=np.float32)
            group_descriptor[group_belongs, index_in_each_group, :] = descriptors[0, select_index, :] * np.expand_dims(weight, axis=1)

            image_descriptor[0, n, v, u, :] = np.sum(group_descriptor, axis=1)[group_belongs, :]

            view_direction[0, n, v, u, :] = np.transpose(point_clouds[0:3, select_index]) - camera_position
            view_direction[0, n, v, u, :] = view_direction[0, n, v, u, :] / (np.tile(np.linalg.norm(view_direction[0, n, v, u, :], axis=1, keepdims=True), (1, 3)) + 1e-10)

            image_output = np.expand_dims(cv2.resize(cv2.imread(image_name, -1), (w, h)), axis=0) / 255.0
            image_descriptor = torch.from_numpy(image_descriptor).permute(0,4,1,2,3).cuda()
            view_direction = torch.from_numpy(view_direction).permute(0,4,1,2,3).cuda()
            if random_crop:

                # limitation of memory etc, we crop the image.
                # Also, we hope crops almost cover the whole image to uniformly optimize point features.
                for j in np.random.permutation(forward_time):
                    movement_v = np.random.randint(0, overlap)
                    movement_u = np.random.randint(0, overlap)

                    if j==0:
                        top_left_u = 0 + movement_u
                        top_left_v = 0 + movement_v
                    if j==1:
                        top_left_u = w_croped - movement_u
                        top_left_v = 0 + movement_v
                    if j==2:
                        top_left_u = 0 + movement_u
                        top_left_v = h_croped - movement_v
                    if j==3:
                        top_left_u = w_croped - movement_u
                        top_left_v = h_croped - movement_v
                    ##Resize image      
                    #image_descriptor = image_descriptor[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)]
                    #view_direction = view_direction[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)]
                    #image_output = image_output[:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :]
                     
                    #image_descriptor = (image_descriptor).permute(0,4,1,2,3)#.cuda()
                    #view_direction = (view_direction).permute(0,4,1,2,3)#.cuda()
                    #print("image descriptor",image_descriptor.shape)
                    #print("view direction", view_direction.shape)
                    opt.zero_grad()

                    data = torch.cat((image_descriptor[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)],view_direction[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)]),1)
                    output = model(data)
                    current_loss = VGG_loss(output[2],torch.from_numpy(image_output[:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :]))[6]
                    print('loss')##Running out of memory here
                    current_loss.backward()
                    print('backprop')
                    opt.step()
                    print('step')                    
                    #input_gradient = torch.autograd.grad(current_loss,image_descriptor)

                    #input_gradient_all[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)] = input_gradient[0] + input_gradient_all[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)]
                    #count[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)] = count[:, :,:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped)] + 1

                #if renew_input:
                #    input_gradient_all = input_gradient_all/(count+1e-10)
                #    descriptors[0, select_index, :] = descriptors[0, select_index, :] - learning_rate_1 * np.expand_dims(descriptor_renew_weight, axis=1) * input_gradient_all[0, n, v, u, :]

            elif not random_crop:
                opt.zero_grad()
                data = torch.cat((image_descriptor,view_direction),1)
                output = model(data)
                current_loss = VGG_loss(output,image_output)
                current_loss.backward()
                opt.step()
                input_gradient = torch.autograd.grad(current_loss,image_descriptor)

                if renew_input:
                    descriptors[0, select_index, :] = descriptors[0, select_index, :] - learning_rate_1 * np.expand_dims(descriptor_renew_weight, axis=1) * input_gradient[0][0, n, v, u, :]
            
            all[i] = current_loss*255.0
            cnt = cnt+1
            print('%s %s %s %.2f %.2f %s' % (epoch, i, cnt, current_loss, np.mean(all[np.where(all)]), time.time() - st))

            if cnt%100==0:
              print('%s/model_pytorch' % (task))
              ##saving at general checkpoint
              torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':opt.state_dict(),
                'loss': current_loss
                          },'%s/model_pytorch' % (task))
              ##saving at epoch specific directory checkpoint
              torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':opt.state_dict(),
            'loss': current_loss
                    },'%s/%04d/model_pytorch' % (task, epoch))

              io.savemat("%s/" % task + 'descriptorpytorch.mat', {'descriptors': descriptors})
            writer.add_scalar('training_loss',current_loss,cnt)
#        os.makedirs("%s/%04d" % (task, epoch))
#        saver.save(sess, "%s/model.ckpt" % (task))
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':opt.state_dict(),
            'loss': current_loss
                    },'%s/%04d/model_pytorch' % (task, epoch))
        io.savemat("%s/" % task + 'descriptorpytorch.mat', {'descriptors': descriptors})

        if epoch % 5 == 0:
#            saver.save(sess, "%s/%04d/model.ckpt" % (task, epoch))
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':opt.state_dict(),
                'loss': current_loss
                        },'%s/%04d/model_pytorch' % (task, epoch))
            io.savemat("%s/%04d/" % (task, epoch) + 'descriptorpytorch.mat', {'descriptors': descriptors})

