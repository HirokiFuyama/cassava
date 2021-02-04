import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=1)

        # Block 1
        self.block0 = Block(channel_in=6, channel_out=64)
        self.block1 = Block(channel_in=64, channel_out=64)
        self.conv2 = nn.Conv2d(64, 128,
                               kernel_size=(1, 1),
                               stride=(2, 2))

        # Block 2
        self.block2 = nn.ModuleList([
            Block(128, 128) for _ in range(2)
        ])

        self.conv3 = nn.Conv2d(128, 256,
                               kernel_size=(1, 1),
                               stride=(2, 2))

        # Block 3
        self.block3 = nn.ModuleList([
            Block(256, 256) for _ in range(2)
        ])

        self.conv4 = nn.Conv2d(256, 512,
                               kernel_size=(1, 1),
                               stride=(2, 2))

        # Block 4
        self.block4 = nn.ModuleList([
            Block(512, 512) for _ in range(3)
        ])

        self.avg_pool = GlobalAvgPool2d()  # TODO: GlobalAvgPool2d
        self.fc = nn.Linear(512, 1000)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.block0(h)
        h = self.block1(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=-1)

        return y

    # def _building_block(self,
    #                     channel_out,
    #                     channel_in=None):
    #     if channel_in is None:
    #         channel_in = channel_out
    #     return Block(channel_in, channel_out)


class Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        # 1x1 の畳み込み
        self.conv1 = nn.Conv2d(channel_in, channel_out,
                               kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu1 = nn.ReLU()

        # 3x3 の畳み込み
        self.conv2 = nn.Conv2d(channel_out, channel_out,
                               kernel_size=(3, 3),
                               padding=0)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.relu2 = nn.ReLU()

        # skip connection用のチャネル数調整
        self.shortcut = self._shortcut(channel_in, channel_out)

        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)  # skip connection
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out,
                         kernel_size=(1, 1),
                         padding=0)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))