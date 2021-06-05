import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

#Upsampling layers 
def upsample_and_concat3D(x1, x2, output_channels, in_channels, only_depth_dims=False):
  if  only_depth_dims:
    deconv = nn.ConvTranspose3d(in_channels,output_channels,kernel_size =(2,1,1),stride=(2,1,1))
    deconv = deconv.cuda()
    return (torch.cat((deconv(x1),x2),1))
  else:
    deconv = nn.ConvTranspose3d(in_channels,output_channels,kernel_size =(2,2,2),stride=(2,2,2))
    deconv = deconv.cuda()
    #print("deconv",deconv(x1).shape)
    print(type(deconv(x1)))
    return (torch.cat((deconv(x1),x2),1))

#Function used to calculate padding, can be used if required to calculate padding, current example has padding hardcoded
def calc_same_padding(in_ch,stride,dilation,kernel):
  p=[0,0,0]
  p[0] = np.ceil(((stride[0]-1)*in_ch[0] + dilation[0]*(kernel[0]-1)+1-stride[0])/2)
  p[1] = np.ceil(((stride[1]-1)*in_ch[1] + dilation[1]*(kernel[1]-1)+1-stride[1])/2)
  p[2] = np.ceil(((stride[2]-1)*in_ch[2] + dilation[2]*(kernel[2]-1)+1-stride[2])/2)
  return tuple(p)

#Main class for UNet
class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv0  = nn.Conv3d(11,16,kernel_size = 1)
    self.conv1  = nn.Conv3d(16,32,kernel_size = 3,padding=1)
    self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2))
    self.conv2  = nn.Conv3d(32,32,kernel_size=3,padding=1)
    self.pool2 = nn.MaxPool3d(kernel_size=(2,1,1),stride=(2,1,1))
    self.conv3  = nn.Conv3d(32,64,kernel_size=3,padding=1)
    self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2))
    self.conv4  = nn.Conv3d(64,64,kernel_size=3,padding=1)
    self.pool4 = nn.MaxPool3d(kernel_size=(2,1,1),stride=(2,1,1))
    self.conv5  = nn.Conv3d(64,128,kernel_size=3,padding=1)
    self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2))
    self.conv6  = nn.Conv3d(128,128,kernel_size=(1,3,3),dilation=(1,2,2),padding=(0,2,2))
    self.conv7 = nn.Conv3d(256,128,kernel_size=3,padding=1)
    self.conv8 = nn.Conv3d(128,64,kernel_size=3,padding=1)
    self.conv9 = nn.Conv3d(128,64,kernel_size=3,padding=1)
    self.conv10 = nn.Conv3d(64,32,kernel_size=3,padding=1)
    self.conv11 = nn.Conv3d(64,32,kernel_size=3,padding=1)
    self.conv12 = nn.Conv3d(32,3,kernel_size=1)
    self.conv12_1 = nn.Conv3d(32,1,kernel_size=1)

  def forward(self,x):
    conv0 = F.leaky_relu(self.conv0(x),0.1)
    print("conv0",conv0.shape)
    conv1 = F.leaky_relu(self.conv1(conv0),0.1)
    print("conv1",conv1.shape)
    pool1 = self.pool1(conv1)
    print("pool1",pool1.shape)
    conv2 = F.leaky_relu(self.conv2(pool1),0.1)
    print("conv2",conv2.shape)
    pool2 = self.pool2(conv2)
    print("pool2",pool2.shape)
    conv3 = F.leaky_relu(self.conv3(pool2),0.1)
    print("conv3",conv3.shape)
    pool3 = self.pool3(conv3)
    print("pool3",pool3.shape)    
    conv4 = F.leaky_relu(self.conv4(pool3),0.1)
    print("conv4",conv4.shape)
    pool4 = self.pool4(conv4)
    print("pool4",pool4.shape)
    conv5 = F.leaky_relu(self.conv5(pool4),0.1)
    print("conv5",conv5.shape)
    pool5 = self.pool5(conv5)
    print("pool5",pool5.shape)
    conv6 = F.leaky_relu(self.conv6((pool5)),0.1)
    print("conv6",conv6.shape)
    up1 = upsample_and_concat3D(conv6,conv5,output_channels=128,in_channels=128,only_depth_dims=False)
    print("up1:",up1.shape)
    conv7 = F.leaky_relu(self.conv7(up1),0.1)
    print("conv7",conv7.shape)
    up2 = upsample_and_concat3D(conv7,conv4,output_channels=64,in_channels=128,only_depth_dims=True)
    print("up2:",up2.shape)
    conv8 = F.leaky_relu(self.conv8(up2),0.1)
    print("conv8",conv8.shape)
    up3 = upsample_and_concat3D(conv8,conv3,output_channels=64,in_channels=64,only_depth_dims=False)
    print("up3:",up3.shape)
    conv9 = F.leaky_relu(self.conv9(up3),0.1)
    print("conv9",conv9.shape)
    up4 = upsample_and_concat3D(conv9,conv2,output_channels=32,in_channels=64,only_depth_dims=True)
    print("up4:",up4.shape)
    conv10 = F.leaky_relu(self.conv10(up4),0.1)
    print("conv10",conv10.shape)
    up5 = upsample_and_concat3D(conv10,conv1,output_channels=32,in_channels=32,only_depth_dims=False)
    print("up5:",up5.shape)
    conv11 = F.leaky_relu(self.conv11(up5),0.1)
    print("conv11",conv11.shape)
    conv12 = F.leaky_relu(self.conv12(conv11),0.1)
    print("conv12",conv12.shape)
    conv12_1 = F.leaky_relu(self.conv12_1(conv11),0.1)
    print("conv12_1",conv12_1.shape)

    weight = torch.nn.functional.softmax(conv12_1,dim=1)
    final = torch.sum(conv12*weight,1)

    return conv12,weight,final

if __name__ == '__main__':
  from torch.autograd import Variable
  data = Variable(torch.rand(1,11,32,480,640))
  data = data.cuda()#conversion to cuda
  net = UNet().cuda()#conversion to cuda
  net(data)#random forward pass