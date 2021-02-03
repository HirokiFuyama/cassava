import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, image_size, channels=3):
        super(CNN, self).__init__()
        self.image_size = image_size

        self.conv1 = nn.Conv2d(channels, 4, kernel_size=8, stride=4, padding=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(4*18*25, 512)

        ####################################################
        # self.fc1 = nn.Linear(64, 512)
        ####################################################

        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        # x = x.view(x.size()[0], -1)
        x = x.reshape(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x