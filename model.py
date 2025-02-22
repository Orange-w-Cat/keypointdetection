import torch.nn as nn
import torch
import torch.nn.functional as F

class conv_vlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_vlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=2, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)
        )

    def forward(self, x, heatmap):
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 双线性插值将尺寸变大
        out = self.layer(up)
        return torch.cat((out, heatmap), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.cov1 = conv_vlock(3, 64)
        self.down1 = Downsample(64)
        self.cov2 = conv_vlock(64, 128)
        self.down2 = Downsample(128)
        self.cov3 = conv_vlock(128, 256)
        self.down3 = Downsample(256)
        self.cov4 = conv_vlock(256, 512)
        self.down4 = Downsample(512)
        self.cov5 = conv_vlock(512, 1024)
        self.up1 = Upsample(1024)
        self.cov6 = conv_vlock(1024, 512)
        self.up2 = Upsample(512)
        self.cov7 = conv_vlock(512, 256)
        self.up3 = Upsample(256)
        self.cov8 = conv_vlock(256, 128)
        self.up4 = Upsample(128)
        self.cov9 = conv_vlock(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1, stride=1)  # 单通道输出，无需 sigmoid

    def forward(self, x):
        R1 = self.cov1(x)
        R2 = self.cov2(self.down1(R1))
        R3 = self.cov3(self.down2(R2))
        R4 = self.cov4(self.down3(R3))
        R5 = self.cov5(self.down4(R4))
        O1 = self.cov6(self.up1(R5, R4))
        O2 = self.cov7(self.up2(O1, R3))
        O3 = self.cov8(self.up3(O2, R2))
        O4 = self.cov9(self.up4(O3, R1))
        return self.out(O4)  # 返回未经过 sigmoid 的输出
