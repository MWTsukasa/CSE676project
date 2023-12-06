import torch
import torch.nn as nn
import torch.nn.functional as F

class FER2013CNN(nn.Module):
    def __init__(self):
        super(FER2013CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 6 * 6)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x