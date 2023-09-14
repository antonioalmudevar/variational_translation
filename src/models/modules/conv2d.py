import torch.nn as nn


class Conv2DBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, p=0.
        ):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x



class Conv2DTransposeBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, p=0.
        ):
        super(Conv2DTransposeBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x