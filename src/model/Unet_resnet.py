import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class Unet_resnet(nn.Module):
    def __init__(self, encoder: str='resnet18', class_num: int=4):
        super(Unet_resnet, self).__init__()

        # Encoder
        if encoder == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            in_channels = (768, 256, 128, 64, 64)
        elif  encoder == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
            in_channels = (768, 256, 128, 64, 64)
        #else:
        #    self.resnet = models.resnet50(pretrained=True)
        #    in_channels = (1024, 512, 256, 128, 64)


        self.layer1 = DecoderBlock(in_channels[0], in_channels[2])
        self.layer2 = DecoderBlock(in_channels[1], in_channels[3])
        self.layer3 = DecoderBlock(in_channels[2], in_channels[4])
        self.layer4 = DecoderBlock(in_channels[2], in_channels[4])
        self.final_conv = nn.Conv2d(in_channels[4], class_num, kernel_size=1)
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

        x = self.layer1(x4, x3)
        x = self.layer2(x, x2)#64, 40, 60
        x = self.layer3(x, x1)#64, 80, 120
        x = self.layer4(x, x0)

        x = self.final_conv(x)
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

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = nn.Sigmoid()
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels // 2,
                                     in_channels // 2,
                                     kernel_size=2,
                                     stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)