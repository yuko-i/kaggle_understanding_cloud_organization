import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet
from torchvision import models
from torchvision.models.resnet import BasicBlock


class Linknet_resnet_ASPP(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super(Linknet_resnet_ASPP, self).__init__()
        # Encoder
        self.resnet_18 = models.resnet18(pretrained=True)
        # Decoder
        self.final_conv = nn.Conv2d(128, 4, kernel_size=(1, 1))
        self.jpu = JointPyramidUpsample([512, 256, 128, 64], 128)
        self.aspp = ASPP(512, 128, rate=[4, 8, 12], dropout_rate=0.1)
        self.up = Up_Pool(512, 128)

        self.initialize()

    def forward(self, x):

        x0 = self.resnet_18.conv1(x)
        x0 = self.resnet_18.bn1(x0)
        x0 = self.resnet_18.relu(x0)

        x1 = self.resnet_18.maxpool(x0)
        x1 = self.resnet_18.layer1(x1)

        x2 = self.resnet_18.layer2(x1)#128, 40, 60
        x3 = self.resnet_18.layer3(x2)#256, 20, 30
        x4 = self.resnet_18.layer4(x3)#512, 40, 60

        x_f = self.jpu([x4, x3, x2, x1])
        x_f = self.aspp(x_f)
        probability_mask = self.final_conv(x_f)
        return probability_mask

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = nn.Softmax(dim=1)
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


class Up_Pool(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_Pool, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.module(x)
        x = F.interpolate(x, size=(320, 480), mode='bilinear', align_corners=True)
        return x


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

        self.attention = SCSEModule(out_channels)

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        x = self.attention(x)

        if skip is not None:
            x = x + skip
            x = self.attention(x)

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


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch // re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch // re, ch, 1),
                                 nn.Sigmoid()
                                 )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ASPPConv(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ASPPConv, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.module(x)
        return x


class ASPPPool(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPPPool, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.module(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel=256, rate=[6, 12, 18], dropout_rate=0.0):
        super(ASPP, self).__init__()

        self.atrous0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.atrous1 = ASPPConv(in_channel, out_channel, rate[0])
        self.atrous2 = ASPPConv(in_channel, out_channel, rate[1])
        self.atrous3 = ASPPConv(in_channel, out_channel, rate[2])
        self.atrous4 = ASPPPool(in_channel, out_channel)

        self.combine = nn.Sequential(
            nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )


    def forward(self, x):
        x = torch.cat([
            self.atrous0(x),
            self.atrous1(x),
            self.atrous2(x),
            self.atrous3(x),
            self.atrous4(x),
        ], 1)
        x = self.combine(x)
        x = F.interpolate(x, size=(320, 480), mode='bilinear', align_corners=True)
        return x


class JointPyramidUpsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JointPyramidUpsample, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel[0], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel[1], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel[2], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel[3], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        # -------------------------------

        self.dilation0 = nn.Sequential(
            SeparableConv2d(4 * out_channel, out_channel, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * out_channel, out_channel, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * out_channel, out_channel, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * out_channel, out_channel, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x0 = self.conv0(x[0])
        x1 = self.conv1(x[1])
        x2 = self.conv2(x[2])
        x3 = self.conv3(x[3])
        x0 = resize_like(x0, x3, mode='nearest')
        x1 = resize_like(x1, x3, mode='nearest')
        x2 = resize_like(x2, x3, mode='nearest')
        x = torch.cat([x0, x1, x2, x3], dim=1)

        d0 = self.dilation0(x)
        d1 = self.dilation1(x)
        d2 = self.dilation2(x)
        d3 = self.dilation3(x)
        x = torch.cat([d0, d1, d2, d3], dim=1)
        return x


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


def resize_like(x, reference, mode='bilinear'):
    if x.shape[2:] != reference.shape[2:]:
        if mode == 'bilinear':
            x = F.interpolate(x, size=reference.shape[2:], mode='bilinear', align_corners=False)
        if mode == 'nearest':
            x = F.interpolate(x, size=reference.shape[2:], mode='nearest')
    return x
