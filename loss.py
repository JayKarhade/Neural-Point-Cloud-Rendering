import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import torchvision.models as models
import torchvision.models.vgg as vgg 
from collections import namedtuple

LossOutput = namedtuple("LossOutput", ["conv1_2", "conv2_2", "conv3_2", "conv4_2","conv5_2"])
#Perceptual Loss Network 
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        #self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '2': "conv1_2",
            '7': "conv2_2",
            '12': "conv3_2",
            '21': "conv4_2",
            '30': "conv5_2"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

def compute_error(real,fake):
  return np.mean(np.abs(fake-real),axis=[1,2,3])

#Compute VGG Loss between real and fake
def VGG_loss(network_output,GT,reuse=False):
    print(GT.shape)
    print(network_output.shape)
    network_output = torch.from_numpy(network_output)
    ##Check dataformat of network output
    # since we use OpenCV to read and write image, bgr to rgb.
    #GT_RGB = torch.cat((GT[:, :, :, 2], GT[:, :, :, 1], GT[:, :, :, 0]), axis=3)
    #network_output_RGB = torch.cat((network_output[:, :, :, 2], network_output[:, :, :, 1], network_output[:, :, :, 0]), axis=3)
    GT = GT.permute(0,3,1,2)
    network_output  = network_output.permute(0,3,1,2)
    fake = network_output*255.0
    real = GT*255.0

    loss_network = LossNetwork()

    real_l = loss_network(real.float())
    fake_l = loss_network(fake.float())

    p = []
    p.append(compute_error(real,fake))
    for i in range(len(fake_l)):
        tmp1 = fake_l[i].cpu().numpy()
        tmp2 = real_l[i].cpu().numpy()
        print("Tmp1",i, np.mean(tmp1), np.std(tmp1))
        print("Tmp2",i,np.mean(tmp2),np.std(tmp2))
        print(compute_error(tmp1,tmp2))
        p.append(compute_error(tmp1,tmp2))
    
    return p[0],p[1],p[2],p[3],p[4],p[5],sum(p)