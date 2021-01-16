import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, image_size, channels=3):
        super(CNN, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.conv1 = nn.Conv2d(self.channels, image_size, kernel_size=4, stride=2, padding=1)
        self.pool = nn.MaxPool2d(10, 10)
        self.conv2 = nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(image_size*2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x