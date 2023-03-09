# -*- coding: utf-8 -*-
# Title     : model_entry.py
# Created by: julse@qq.com
# Created on: 2022/7/13 18:56
# des : 模型首行转换一次维度
from tkinter import Variable

import torch
import torchsummary
from torch import nn, Tensor

from model._0713 import mingpt
from model._0713.resnet import BasicBlock



class WrapLayers(nn.Sequential):
    def __init__(self,*args: nn.Module):
        super(WrapLayers, self).__init__(*args)
    def forward(self, input: Tensor, mask: Tensor = None,*args,**kwargs) -> Tensor:
        # x = self.layers(x,mask=mask,hello='')
        for module in self:
            input = module(input, mask=mask,*args,**kwargs)
            # if isinstance(module,(BasicBlock,Block,WrapLayers)):
            #     input = module(input, mask=mask)
            # else:input = module(input)
            # # if isinstance(module,nn.Dropout):input = module(input)
            # # else: input = module(input,mask=mask)
            # # input = module(input, mask=mask)
        return input
class FrameModel(nn.Sequential):
    def __init__(self,in_channels,out_channels,planes=64,depth=3,model_type='senet',mask=True,keepdim=4,reduce2dim=4,norm_layer_type='BATCHINSTANCENORM1D',net_block=mingpt.Block,*args,**kwargs):
        super(FrameModel, self).__init__(*args,**kwargs)
        norm_layer_dict = {
            "BATCHNORM1D":nn.BatchNorm1d,
        }
        if norm_layer_type.upper() in norm_layer_dict:
            norm_layer = norm_layer_dict[norm_layer_type.upper()]
        else:exit('no such normal style'+ self.norm_layer_type)

        self.in_planes = in_channels
        self.mask = mask
        self.keepdim = keepdim
        self.reducedim = None
        self.build_net(model_type,planes,norm_layer,depth,net_block=net_block)
        self.final = nn.Linear(planes,out_channels)
    def build_net(self,model_type,planes,norm_layer,depth,net_block=mingpt.Block):
        self.net = WrapLayers(self._make_layer(BasicBlock, planes, 1, stride=1, norm_layer=norm_layer),
                              self._make_layer(mingpt.Block, planes, depth, stride=1))
        self.in_planes = planes
    def forward(self, x: Tensor) -> Tensor:
        # compress dim
        # if isinstance(self.net,nn.Conv1d) or isinstance(self.net,UNet):x = self.net(x)
        # if not self.mask:
        if self.reducedim!=None:
            emb = x[:, self.keepdim:-1, :]
            emb = emb.permute(0, 2, 1)
            emb = self.reducedim(emb)
            emb = emb.permute(0, 2, 1)
            x = torch.concat([x[:, :self.keepdim, :],emb], 1)
            x = self.net(x)
        else:
            line_mask = x[:, -1, :]  # [8, L]
            line_mask = line_mask.unsqueeze(1)
            # padding_mask = line_mask.repeat(1,x.size()[1],1)
            # padding_mask = line_mask.repeat(x.size()[1], 1, 1)  # [channels, 8, L]
            # padding_mask = padding_mask.permute(1, 0, 2)
            # x = x[:, :-1, :]  # [8, 64, L]
            # x = x.to(torch.float32)

            if self.reducedim:
                emb = x[:, self.keepdim:-1, :]
                # emb = emb.permute(0, 2, 1)
                emb = self.reducedim(emb)
                # emb = emb.permute(0, 2, 1)
                x = torch.concat([x[:, :self.keepdim, :], emb,line_mask], 1)

            # x = F.dropout(F.elu(self.inibn(self.iniconv(x))),0.4)
            # x = torch.mul(line_mask, x)
            x = self.net(x,mask=line_mask) # input x [16, 5, 340]

        x = x.permute(0, 2, 1)
        x = self.final(x)
        x = torch.sigmoid(x) # == F.sigmoid(x) 只因为后者被废弃，用前者替代
        return x

    def _make_layer(self, block, planes, num_blocks,stride=1,dilated=True,*args,**kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        if not dilated:dilation = [0] + [0] * (num_blocks - 1)
        else:dilation = list(range(num_blocks))
        # print(strides)
        layers = []
        for i,stride in enumerate(strides):
            # print(planes,stride)
            # layers.append(block(self.in_planes, planes,stride,dilation=dilation[i]))
            layers.append(block(self.in_planes, planes,stride,*args,**kwargs))
            self.in_planes = planes

            # layers.append(nn.Dropout(0.4))
        return WrapLayers(*layers)

class MaksLearn(nn.Module):
    def __init__(self):
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass

class MultiViewModel(nn.Module):
    def __init__(self,models):
        self
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass

import time

# if __name__ == '__main__':
#     print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
#     start = time.time()
#     in_channels=5
#     out_channels=18
#     dim = 64
#     depth = 3
#     mask = True
#     modeltype = 'senet'
#     model = rsaModel(4,18,depth=3)
#     model =BasicBlock(in_channels,dim)
#     print(model)
#     x = torch.from_numpy(np.random.random((1, in_channels, 41)))
#     out = model(x)
#     out = torch.squeeze(out)
#
#     torchsummary.summary(model,np.random.random((1, in_channels, 41)))
#     out = torch.argmax(out,dim=0)
#     print("finally outshape:", out.shape)
#
#
#     pass
#     print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
#     print('time', time.time() - start)

