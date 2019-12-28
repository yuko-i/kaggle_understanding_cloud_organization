import torch.nn as nn
import torch
from torchvision import models


class Linknet_resnet(nn.Module):
    def __init__(self, encoder: str='resnet18', class_num: int=4):
        super(Linknet_resnet, self).__init__()

        # Encoder
        if encoder == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif  encoder == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)

        # Decoder
        in_channels = (512, 256, 128, 64, 64)
        self.layer1 = DecoderBlock(in_channels[0], in_channels[1])
        self.layer2 = DecoderBlock(in_channels[1], in_channels[2])
        self.layer3 = DecoderBlock(in_channels[2], in_channels[3])
        self.layer4 = DecoderBlock(in_channels[3], in_channels[4])
        self.layer5 = DecoderBlock(in_channels[4], 32)
        self.final_conv = nn.Conv2d(32, class_num, kernel_size=(1, 1))
        self.initialize()

    def forward(self, x):

        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)

        x1 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x1)

        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        ec_x = [x4, x3, x2, x1, x0]
        encoder_head = ec_x[0]
        skips = ec_x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)
        return x

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