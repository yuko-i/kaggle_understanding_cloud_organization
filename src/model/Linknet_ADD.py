import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet
from torchvision import models
from torchvision.models.resnet import BasicBlock

class Linknet_resnet18(nn.Module):
    def __init__(self, n_classes=4):
        super(Linknet_resnet18, self).__init__()

        # Encoder
        self.resnet_18 = models.resnet18(pretrained=True)
        # Decoder
        in_channels = (512, 256, 128, 64, 64)
        self.layer1 = DecoderBlock(in_channels[0], in_channels[1])
        self.layer2 = DecoderBlock(in_channels[1], in_channels[2])
        self.layer3 = DecoderBlock(in_channels[2], in_channels[3])
        self.layer4 = DecoderBlock(in_channels[3], in_channels[4])
        self.layer5 = DecoderBlock(in_channels[4], 32)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=(1, 1))
        self.separable = SeparableConv2d(32, 4)

        self.initialize()

    def forward(self, x):

        x0 = self.resnet_18.conv1(x)
        x0 = self.resnet_18.bn1(x0)
        x0 = self.resnet_18.relu(x0)

        x1 = self.resnet_18.maxpool(x0)
        x1 = self.resnet_18.layer1(x1)

        x2 = self.resnet_18.layer2(x1)
        x3 = self.resnet_18.layer3(x2)
        x4 = self.resnet_18.layer4(x3)

        ec_x = [x4, x3, x2, x1, x0]
        encoder_head = ec_x[0]
        skips = ec_x[1:]

        xd = self.layer1([encoder_head, skips[0]])
        xd = self.layer2([xd, skips[1]])
        xd = self.layer3([xd, skips[2]])
        xd = self.layer4([xd, skips[3]])
        xd = self.layer5([xd, None])
        probability_mask = self.separable(xd)
        #probability_mask = self.final_conv(xd)
        return probability_mask

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = nn.Sigmoid()
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class TransposeX2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, in_channels // 4, kernel_size=1),
            TransposeX2(in_channels // 4, in_channels // 4),
            Conv2dReLU(in_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x, skip = x
        x = self.block(x)

        if skip is not None:
            x = x + skip

        return x

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, dilation, groups=in_channel,
                              bias=bias)
        self.bn = nn.BatchNorm2d(in_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x