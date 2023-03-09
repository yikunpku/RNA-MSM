import torch
from torch import nn
from torch.nn import functional as F


"""New Resdule"""
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1,dropout=0.4,norm_layer=nn.BatchNorm1d,*args,**kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.dr1 = nn.Dropout(dropout,inplace=False)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        # nn.init.constant_(self.bn2.weight, 0)
        self.dr2 = nn.Dropout(dropout, inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

        # self.fc1 = nn.Linear(planes, planes//16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        # self.fc2 = nn.Linear(planes//16, planes, kernel_size=1)
    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = self.dr1(F.relu(self.bn1(self.conv1(x))))
        out = torch.mul(mask,out)
        out = self.dr2(F.relu(self.bn2(self.conv2(out))))
        out = torch.mul(mask,out)

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        out = out * w  # New broadcasting feature_connection from v0.2!
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SENET15(nn.Module):
    def __init__(self, in_planes, planes, stride=1,dilation=[1,2]):
        super(SENET15, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=15, stride=stride, dilation=dilation[0],padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=15, stride=1, dilation=dilation[1],padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

        # self.fc1 = nn.Linear(planes, planes//16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        # self.fc2 = nn.Linear(planes//16, planes, kernel_size=1)
    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = F.dropout(F.relu(self.bn1(self.conv1(x))),p=0.4)
        out = torch.mul(mask,out)
        out = F.dropout(F.relu(self.bn2(self.conv2(out))),p=0.4)
        out = torch.mul(mask,out)

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        out = out * w  # New broadcasting feature_connection from v0.2!
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SenetOri(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(SenetOri, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

        # SE layers
        # self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        # self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

        self.fc1 = nn.Linear(planes, planes//16)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Linear(planes//16, planes)
    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = F.dropout(F.relu(self.bn1(self.conv1(x))),p=0.4)
        out = torch.mul(mask,out)
        out = F.dropout(F.relu(self.bn2(self.conv2(out))),p=0.4)
        out = torch.mul(mask,out)

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = w.permute(0, 2, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        w = w.permute(0, 2, 1)
        out = out * w  # New broadcasting feature_connection from v0.2!
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Snap2Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Snap2Block, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,  padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=5, stride=1,  padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = F.dropout(F.elu(self.bn1(self.conv1(x))),p=0.4)
        out = torch.mul(mask,out)
        out = F.dropout(F.elu(self.bn2(self.conv2(out))),p=0.4)
        out = torch.mul(mask,out)
        out += self.shortcut(x)
        out = F.elu(out)
        return out
class Spot1Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Spot1Block, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=5, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1,  padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, padding='same', bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = F.dropout(F.elu(self.bn1(self.conv1(x))),p=0.4)
        out = torch.mul(mask,out)
        out = F.dropout(F.elu(self.bn2(self.conv2(out))),p=0.4)
        out = torch.mul(mask,out)
        out += self.shortcut(x)
        out = F.elu(out)
        return out


'''todo 反卷积尺寸不匹配'''
class CnntrBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(CnntrBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_planes, planes, kernel_size=5, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(planes, planes, kernel_size=7, stride=1,  padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, padding='same', bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = F.elu(self.bn1(self.conv1(x)))
        out = torch.mul(mask,out)
        out = F.elu(self.bn2(self.conv2(out)))
        out = torch.mul(mask,out)
        out += self.shortcut(x)
        out = F.elu(out)
        return out

class CnntrmixBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(CnntrmixBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=5, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(planes, planes, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, padding='same', bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x,mask=None):
        # x = x.to(torch.float32)
        out = F.elu(self.bn1(self.conv1(x)))
        out = torch.mul(mask,out)
        out = F.elu(self.bn2(self.conv2(out)))
        out = torch.mul(mask,out)
        out += self.shortcut(x)
        out = F.elu(out)
        return out
'''todo'''
class SENet(nn.Module):
    def __init__(self, block, num_blocks, input_channel):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1)
        # self.linear = nn.Linear(512, num_classes)
        self.cnn = CNNnet()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.cnn(out)
        # print(out.shape)
        return out


# 定义网络结构 初始输入为64,用来做为decoder层
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),

            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 16, 3, 1, 1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 1, 3, 1, 1),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        return x

def SENet18(input):
    return SENet(BasicBlock, [3, 3, 2, 2], input)


# def test():
#     se_net = SENet18()
#     y = se_net(torch.randn(1, 3, 32, 32))
#     cnn = CNNnet()
#     out = cnn(y)
#     print(out.shape)
# test()


# def main1():
#     # blk = ResBlk(64, 128)  # 分别给ch_in, ch_out赋值
#     # tmp = torch.randn(2, 64, 500, 500)  # 为什么有四个参数？
#     # out = blk(tmp)
#     # print('block:', out.shape)
#
#     model = ResNet18()  # num_class = 5
#     tmp = torch.randn(2, 2, 700, 700)
#
#     out = model(tmp)
#     print("resnet:", out.shape)
#     p = sum(map(lambda p:p.numel(), model.parameters()))   #其中的：lambda p:p.numel()，是什么意思？
#     print('parameters size', p)

def main():
    # blk = ResBlk(64, 128)  # 分别给ch_in, ch_out赋值
    # tmp = torch.randn(2, 64, 500, 500)  # 为什么有四个参数？
    # out = blk(tmp)
    # print('block:', out.shape)

    model = SENet18()  # num_class = 5
    tmp = torch.randn(2, 2, 700, 700)

    out = model(tmp)
    print("resnet:", out.shape)
    p = sum(map(lambda p:p.numel(), model.parameters()))   #其中的：lambda p:p.numel()，是什么意思？
    print('parameters size', p)

if __name__ == '__main__':
    # main()
    tmp = torch.randn(2, 5, 700)
    model = SENet(BasicBlock,[2,3,4,5],5)
    model(tmp)
    print(model)

#     print(model(tmp))