import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Simple3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(3840, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # flatten
        x = x.view(-1, 3840)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class ThreeSliceCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ThreeSliceCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Only pool across height and width
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))

        # flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x