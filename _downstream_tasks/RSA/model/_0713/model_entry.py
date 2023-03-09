# -*- coding: utf-8 -*-
# Title     : model_entry.py
# Created by: julse@qq.com
# Created on: 2022/7/13 18:56
# des : TODO
from tkinter import Variable

import torch
import torchsummary
from msm.modules import TransformerLayer
from torch import nn, Tensor

from model._0713.resnet import BasicBlock, SENet, Snap2Block, Spot1Block, CnntrBlock, CnntrmixBlock, SenetOri
from model._0713.senet2 import SENET2
from model._0713.unet import UNet

# from resnet import BasicBlock
# from unet import UNet
import numpy as np
import torch.nn.functional as F

# def martchModel(in_channels,out_channels,depth=3,dim=64,mask=True,model='senet',*args):
#     if model =='senet':
#         midnet = BasicBlock(in_channels,dim)
#
#
# class rsaModel(nn.Sequential):
#     def __init__(self,in_channels,out_channels,depth=3,dim=64,mask=True,model='senet',*args):
#         super(rsaModel, self).__init__(*args)
#         # self.unet = UNet(out_channels, in_channels=in_channels, depth=4, merge_mode='concat')
#         # self.net = nn.Linear(in_channels, dim)
#         # if model=='senet':
#         self.unet = BasicBlock(in_channels,dim)
#         self.linear = nn.Linear(dim,18)
#     def forward(self, x: Tensor) -> Tensor:
#         # line_mask = x[:, -1, :]  # [8, L]
#         # padding_mask = line_mask.repeat(self.inplanes, 1, 1)  # [channels, 8, L]
#         # padding_mask = padding_mask.permute(1, 0, 2)
#         # x = x[:, :-1, :]  # [8, 64, L]
#         x = self.unet(x)
#         x = x.permute(0, 2, 1)
#         x = self.linear(x)
#         x = torch.sigmoid(x)
#         return x
from model.model_snap2_4 import RRI_Model


class WrapLayers(nn.Sequential):
    def __init__(self,*args: nn.Module):
        super(WrapLayers, self).__init__(*args)
    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        # x = self.layers(x,mask=mask,hello='')
        for module in self:
            # if isinstance(module,nn.Dropout):input = module(input)
            # else: input = module(input,mask=mask)
            input = module(input, mask=mask)

        return input
class FrameModel(nn.Sequential):
    def __init__(self,in_channels,out_channels,planes=64,depth=3,model_type='senet',mask=True,norm_layer_type='',*args):
        super(FrameModel, self).__init__(*args)
        self.in_planes = in_channels
        self.mask = mask
        if model_type =='SENET':self.net = self._make_layer(BasicBlock,planes,depth,stride=1)
        elif model_type =='SENETORI':self.net = self._make_layer(SenetOri,planes,depth,stride=1)
        elif model_type =='SNAP2':self.net = self._make_layer(Snap2Block,planes,depth,stride=1)
        elif model_type =='SPOT1':self.net = self._make_layer(Spot1Block,planes,depth,stride=1)
        elif model_type =='TRANSFORMER':self.net = self._make_layer(TransformerLayer,planes,depth,stride=1) # stride is attention head  /embed_dim must be divisible by num_heads
        elif model_type =='UNET':
            self.net = UNet(dim,in_channels,depth=depth)
            self.mask = False
        elif model_type == 'SENET2':self.net = SENET2(in_channels, planes,depth,out_channels)
        elif model_type =='CnntrBlock':self.net = self._make_layer(CnntrBlock,planes,depth,stride=1)
        elif model_type =='CnntrmixBlock':self.net = self._make_layer(CnntrmixBlock,planes,depth,stride=1)
        elif model_type[:4] =='CNN_' :
            layers = []
            for i in range(depth):
                layers.append(nn.Conv1d(self.in_planes,planes,kernel_size=(eval(model_type.split('_')[1]),),padding='same'))
                self.in_planes = planes
            # self.net = nn.Sequential(*layers)
            self.net = WrapLayers(*layers)
            self.mask=False
        elif model_type =='snap2_4':self.net = RRI_Model(bbn=depth,myChannels = 1024,inplanes=64,dilation=1,blocktype=Snap2Block,dropout=0.4,norm_layer_type='BATCHNORM1D',conv_mask = False,zero_init_residual=True,affine=True)
        # else:self.net = nn.Conv1d(in_channels,dim,kernel_size=(5,),padding='same')
        else:exit('no such model',model_type)
        # self.linear = nn.Linear(dim,18)
        self.final = nn.Linear(planes,out_channels)
    def forward(self, x: Tensor) -> Tensor:
        # if isinstance(self.net,nn.Conv1d) or isinstance(self.net,UNet):x = self.net(x)
        if not self.mask:x = self.net(x)
        else:
            line_mask = x[:, -1, :]  # [8, L]
            line_mask = line_mask.unsqueeze(1)
            # padding_mask = line_mask.repeat(1,x.size()[1],1)
            # padding_mask = line_mask.repeat(x.size()[1], 1, 1)  # [channels, 8, L]
            # padding_mask = padding_mask.permute(1, 0, 2)
            # x = x[:, :-1, :]  # [8, 64, L]
            # x = x.to(torch.float32)
            x = torch.mul(line_mask, x)
            x = self.net(x,mask=line_mask)

        x = x.permute(0, 2, 1)
        x = self.final(x)
        x = torch.sigmoid(x)
        return x


    def _make_layer(self, block, planes, num_blocks,stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        print(strides)
        layers = []
        for stride in strides:
            print(planes,stride)
            layers.append(block(self.in_planes, planes,stride))
            # layers.append(nn.Dropout(0.4))
            self.in_planes = planes
        return WrapLayers(*layers)

import time

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    in_channels=5
    out_channels=18
    dim = 64
    depth = 3
    mask = True
    modeltype = 'senet'
    model = rsaModel(4,18,depth=3)
    model =BasicBlock(in_channels,dim)
    print(model)
    x = torch.from_numpy(np.random.random((1, in_channels, 41)))
    out = model(x)
    out = torch.squeeze(out)

    torchsummary.summary(model,np.random.random((1, in_channels, 41)))
    out = torch.argmax(out,dim=0)
    print("finally outshape:", out.shape)


    pass
    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)

