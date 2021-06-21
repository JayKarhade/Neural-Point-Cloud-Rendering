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
  real = real.cuda()
  fake = fake.cuda()
  return torch.mean(torch.abs(fake-real),axis=[1,2,3]).cuda()

#Compute VGG Loss between real and fake
def VGG_loss(network_output,GT,reuse=False):
    #print(GT.shape)
    #print(network_output.shape)
    #network_output = torch.from_numpy(network_output)
    ##Check dataformat of network output
    # since we use OpenCV to read and write image, bgr to rgb.
    #GT_RGB = torch.cat((GT[:, :, :, 2], GT[:, :, :, 1], GT[:, :, :, 0]), axis=3)
    #network_output_RGB = torch.cat((network_output[:, :, :, 2], network_output[:, :, :, 1], network_output[:, :, :, 0]), axis=3)
    GT = GT.permute(0,3,1,2)
    #network_output  = network_output.permute(0,3,1,2)
    fake = network_output*255.0
    real = GT*255.0
    #print('c1')
    loss_network = LossNetwork().cuda()
    #print('c2')
    real_l = loss_network(real.float().cuda())
    fake_l = loss_network(fake.float().cuda())
    #print('c3')
    p = []
    p.append(compute_error(real.cuda(),fake.cuda()))
    for i in range(len(fake_l)):
        tmp1 = fake_l[i]
        tmp2 = real_l[i]
        #print("Tmp1",i, torch.mean(tmp1), torch.std(tmp1))
        #print("Tmp2",i,torch.mean(tmp2),torch.std(tmp2))
        #print(compute_error(tmp1.cuda(),tmp2.cuda()))
        p.append(compute_error(tmp1.cuda(),tmp2.cuda()))
    #print("total loss",sum(p))
    return p[0],p[1],p[2],p[3],p[4],p[5],sum(p)

if __name__ == '__main__':
  x1 = torch.rand(1,480,640,3)
  x2 = torch.rand(1,3,480,640)
  z = VGG_loss(x2,x1)[6]
  print(z)