import torch
import torch.nn as nn
import os
from utils.net_util import weight_initialization
from model.backbone.layer_factory import BottleNeck, ConvLayer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):
    def __init__(self, block, layers,
                 planes=(16, 32, 64, 128, 256, 512, 512, 512)):
        super(DRN, self).__init__()
        self.in_dim = planes[0]
        self.out_dim = planes[-1]
        self.layer0 = []
        self.layer0 += [ConvLayer(in_dim=3, out_dim=planes[0], kernel_size=7, padding=3,
                                  use_bias=False, norm='bn', activation='relu')]
        self.layer0 = nn.Sequential(*self.layer0)
        self.layer1 = self._make_conv_layers(planes[0], layers[0], stride=1)
        self.layer2 = self._make_conv_layers(planes[1], layers[1], stride=2)

        self.layer3 = self._make_layers(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layers(block, planes[3], layers[3], stride=2)
        self.layer5 = self._make_layers(block, planes[4], layers[4], dilation=2,
                                        new_level=False)
        self.layer6 = self._make_layers(block, planes[5], layers[5], dilation=4,
                                        new_level=False)

        self.layer7 = self._make_conv_layers(planes[6], layers[6], dilation=2)
        self.layer8 = self._make_conv_layers(planes[7], layers[7], dilation=1)

    def _make_conv_layers(self, dim, convs, stride=1, dilation=1):
        layers = []
        for i in range(convs):
            layers += [ConvLayer(in_dim=self.in_dim, out_dim=dim, kernel_size=3,
                                 stride=stride if i == 0 else 1,
                                 padding=dilation, use_bias=False, dilation=dilation,
                                 norm='bn', activation='relu')]
            self.in_dim = dim
        return nn.Sequential(*layers)

    def _make_layers(self, block, dim, num_blocks, stride=1, dilation=1,
                     new_level=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.in_dim != dim * block.expansion:
            downsample = nn.Sequential(ConvLayer(in_dim=self.in_dim, out_dim=dim * block.expansion, kernel_size=1,
                                                 stride=stride, norm='bn', use_bias=False))
            # set_require_grad(downsample._modules['0'].norm, False)
        layers = []

        layers += [block(in_dim=self.in_dim, out_dim=dim, stride=stride, downsample=downsample,
                         dilation=(1, 1) if dilation == 1 else (
                             dilation // 2 if new_level else dilation, dilation)
                         )]

        self.in_dim = dim * block.expansion
        for i in range(num_blocks - 1):
            layers += [block(in_dim=self.in_dim, out_dim=dim, dilation=(dilation, dilation))]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        low_level_feature = out
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out, low_level_feature


def drn_d_105(pre_trained=True):
    """
    drn_d_105
    """
    drn = DRN(BottleNeck, [1, 1, 3, 4, 23, 3, 1, 1])
    if pre_trained:
        pretrained_dict = torch.load('../pretrained_models/drn_d_105.pth')
        drn.load_state_dict(pretrained_dict)
    else:
        drn = weight_initialization(drn)
    return drn

if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    drn = drn_d_105(pre_trained=True)
    drn = drn.to(device)
    image = torch.ones(1, 3, 700, 1280)
    image = image.to(device)
    output, low_level_feature = drn(image)
    print(output)
    print(output.size())

