import os.path
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class LinearModule(nn.Module):
    """A class represent a simple NN for digitalnumber MNIST"""

    def __init__(self, input_size, hidden_layer_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear_1 = nn.Linear(self.input_size, hidden_layer_size)
        self.linear_2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.linear_3 = nn.Linear(hidden_layer_size // 2, 10)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flattern
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        x = self.linear_3(x)
        return F.log_softmax(x, dim=1)


class ConvModule(nn.Module):
    def __init__(self, fully_connected_size, convolutional_size):
        super(ConvModule, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, convolutional_size, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(convolutional_size, convolutional_size * 2, kernel_size=3, stride=1)

        # Define the fully connected layers
        self.fc1 = nn.Linear(convolutional_size * 2 * 5 * 5, fully_connected_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fully_connected_size, 10)

    def forward(self, x):
        # Apply the convolutional layers
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)

        # Flatten the output from the convolutional layers
        x = torch.flatten(x, 1)

        # Apply the fully connected layers
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x