import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from cbam import *
# from SEResNext import *

# SENet模块实现
class SE_module(nn.Module):
    def __init__(self,channel=None, r=16):
        super(SE_module,self).__init__()

        self._avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self._fc = nn.Sequential(
            nn.Conv2d(channel, channel//r,1,bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        y = self._avg_pool(x)
        y = self._fc(y)
        return x*y

# Channel Attention Module 模块实现
class Channel_Attention(nn.Module):
    def __init__(self, channel=None, r=16):
        super(Channel_Attention,self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self._fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
        )
        self._sigmoid = nn.Sigmoid()

    def forward(self,x):
        y1 = self._avg_pool(x)
        y1 = self._fc(y1)

        y2 = self._max_pool(x)
        y2 = self._fc(y2)

        y = self._sigmoid(y1+y2)

        return x*y


# Spatial Attention Moudlue实现
class Spatial_Attention(nn.Module):
    def __init__(self,kernel_size=3):
        super(Spatial_Attention, self).__init__()
        # kernel_size 一定得是奇数，要不不能满足feature map尺寸不变
        assert kernel_size %2 ==1,"kernel_size={}".format(kernel_size)
        padding = (kernel_size-1)//2
        self._layer = nn.Sequential(
            nn.Conv2d(2,1,kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self,x):
        avg_mask = torch.mean(x,dim=1,keepdim=True)
        max_mask,_ = torch.max(x,dim=1,keepdim=True)
        mask = torch.cat([avg_mask,max_mask],dim=1)

        mask = self._layer(mask)
        return x*mask

class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for i in self.num_levels:
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu1 = nn.LeakyReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        # self.relu2 = nn.ReLU(True)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        # y = self.relu2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True) # 残差连接

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
            # blocks+= [SE_module(c_out)]
        else:
            blocks += [BasicBlock(c_out,c_out),]
            # blocks += [SE_module(c_out)]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64,64,2,False)
        # 32 64 32
        self.layer2 = make_layers(64,128,2,True)
        # 64 32 16
        self.layer3 = make_layers(128,256,2,True)
        # 128 16 8
        self.layer4 = make_layers(256,512,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


class SEResidualNet(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(SEResidualNet, self).__init__()
        # output = (input -ks +2*p)/s +1
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 64 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # se1
        self.se1 = SE_module(64)
        # 128 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # se2
        self.se2 = SE_module(128)
        # 256 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # se3
        self.se3 = SE_module(256)
        # 512 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # se4
        self.se4 = SE_module(512)
        # 512 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 512 1 1
        self.reid = reid

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

class ChannelResidualNet(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(ChannelResidualNet, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        self.ca1 = Channel_Attention(64)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        self.ca2 = Channel_Attention(128)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        self.ca3 = Channel_Attention(256)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        self.ca4 = Channel_Attention(512)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.ca1(x)
        x = self.layer2(x)
        x = self.ca2(x)
        x = self.layer3(x)
        x = self.ca3(x)
        x = self.layer4(x)
        x = self.ca4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

class SpatialResidualNet(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(SpatialResidualNet, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        self.sa1 = Spatial_Attention()
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        self.sa2 = Spatial_Attention()
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        self.sa3 = Spatial_Attention()
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        self.sa4 = Spatial_Attention()
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.sa1(x)
        x = self.layer2(x)
        x = self.sa2(x)
        x = self.layer3(x)
        x = self.sa3(x)
        x = self.layer4(x)
        x = self.sa4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

class CBAMResidualNet(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(CBAMResidualNet, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        self.cbam1 = CBAM(64)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        self.cbam2 = CBAM(128)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        self.cbam3 = CBAM(256)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        self.cbam4 = CBAM(512)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.cbam1(x)
        x = self.layer2(x)
        x = self.cbam2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = Net().to(device)
    # net2 = SEResidualNet().to(device)
    # net3 = ChannelResidualNet().to(device)
    # net4 = SpatialResidualNet().to(device)
    net5 = CBAMResidualNet().to(device)
    net6 = resnext101_32x8d(751).to(device)
    x = torch.randn(4,3,128,64).to(device)
    print(x.device)
    # y = net(x)
    # z = net2(x)
    # k = net3(x)
    # l = net4(x)
    # j = net5(x)
    m = net6(x)
    from torchsummary import summary
    # summary(net,(3,128,64))
    # summary(net2, (3, 128, 64))
    # summary(net3, (3, 128, 64))
    # summary(net4,(3,128,64))
    summary(net6, (3, 128, 64))
    # import ipdb; ipdb.set_trace()


