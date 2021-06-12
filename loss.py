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
    ##Check dataformat of network output
    # since we use OpenCV to read and write image, bgr to rgb.
    network_output_RGB = torch.cat([network_output[:, :, :, 2:3], network_output[:, :, :, 1:2], network_output[:, :, :, 0:1]], axis=3)
    GT_RGB = torch.cat([GT[:, :, :, 2:3], GT[:, :, :, 1:2], GT[:, :, :, 0:1]], axis=3)

    fake = network_output_RGB*255.0
    real = GT_RGB*255.0

    loss_network = LossNetwork()

    fake_l = LossNetwork(fake)
    real_l = LossNetwork(real)

    p = []

    for i in range(len(fake_l)):
        tmp1 = fake_l[i].cpu().numpy()
        tmp2 = real_l[i].cpu().numpy()
        print("Tmp1",i, np.mean(tmp1), np.std(tmp1))
        print("Tmp2",i,np.mean(tmp2),np.std(tmp2))
        print(compute_error(tmp1,tmp2))
        p.append(compute_error(tmp1,tmp2))
    
    return p[0],p[1],p[2],p[3],p[4],sum(p)