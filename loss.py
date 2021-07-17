import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import torchvision.models as models
import torchvision.models.vgg as vgg 
from collections import namedtuple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
  real = real.to(device)
  fake = fake.to(device)
  return torch.mean(torch.abs(fake-real),dim=(1,2,3))

#Compute VGG Loss between real and fake
def VGG_loss(network_output,GT,reuse=False):
    #print(GT.shape)
    #print(network_output.shape)
    #network_output = torch.from_numpy(network_output)
    ##Check dataformat of network output
    m = torch.tensor([0.485, 0.456, 0.406]).to(device)*255.0
    m = torch.reshape(m,(1,3,1,1)).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)*255.0
    std = torch.reshape(std,(1,3,1,1)).to(device)
    GT = GT.permute(0,3,1,2).to(device) 
    # since we use OpenCV to read and write image, bgr to rgb.
    #GT_RGB = torch.cat((GT[:, 2, :, :], GT[:, 1, :, :], GT[:, 0, :, :]),axis=2)
    #network_output_RGB = torch.cat((network_output[:, 2, :,:], network_output[:, 1, :, :], network_output[:, 0, :, :]),axis=2)
    
    #network_output  = network_output.permute(0,3,1,2)
    fake = ((network_output*255.0))# - m)/std
    real = ((GT*255.0))# - m)/std
    #print('c1')
    loss_network = LossNetwork().to(device)
    #print('c2')
    real_l = loss_network(real.float().to(device))
    fake_l = loss_network(fake.float().to(device))
    #print('c3')
    p = []
    div_nums = [1.6,2.3,1.8,2.8,12.5]##Constant used to divide losses
    p.append(compute_error(real.to(device),fake.to(device)))
    for i in range(len(fake_l)):
        tmp1 = fake_l[i]
        tmp2 = real_l[i]
        p.append(compute_error(tmp1.to(device),tmp2.to(device))/div_nums[i])
    #print("total loss",sum(p))
    return p[0],p[1],p[2],p[3],p[4],p[5],sum(p)

if __name__ == '__main__':
  x1 = torch.rand(1,480,640,3).to(device)
  #x1 = torch.rand(1,3,480,640)  
  x2 = torch.rand(1,3,480,640).to(device)
  #print(compute_error(x1,x2))
  z = VGG_loss(x2,x1)
  print(z)
