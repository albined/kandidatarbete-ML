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


class linear_module(nn.Module):
    """A class represent a simple NN for digitalnumber MNIST"""

    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(28 * 28, 254)
        self.linear_2 = nn.Linear(254, 64)
        self.linear_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flattern
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        x = torch.relu(self.linear_3(x))  # softmax då vi har fler än 2 klasser (sigmoid om man har 2 klasser)
        return x


class conv_module(nn.Module):
    """A class represent a more advanced NN for digitalnumber MNIST"""

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(2)
        self.linear_1 = nn.Linear(64 * 7 * 7, 128)
        self.linear_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv_1(x))  # outputsize (28x28x32)
        x = self.maxpool(x)  # outputsize (14x14x32)
        x = torch.relu(self.conv_2(x))  # outputsize (14x14x64)
        x = self.maxpool(x)  # outputsize (7x7x64)
        x = x.view(-1, 64 * 7 * 7)  # Flattern the data outputsize: (1x1x64*7*7)
        x = torch.relu(self.linear_1(x))  # outputsize (128)
        x = torch.relu(self.linear_2(x))  # outputsize (10)
        return x


class conv_module_general(nn.Module):
    """A class represent a more advanced and general NN"""

    def __init__(self, channels: int, matrix_shape: list) -> None:
        super().__init__()
        # Defines varibles
        out_chanel_2 = 64
        n_maxpool = 2
        # Convulutional layer
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=out_chanel_2, kernel_size=3, stride=1, padding=1)
        # Pooling layer
        self.maxpool = nn.MaxPool2d(n_maxpool)
        # dropout with 50% chance of dropout
        # self.dropout = nn.Dropout(p=0.1)
        # Calculate the input to the linearlayer depending on the input to the NN,
        # NOTE: only out_chanel_2,  n_maxpool and input chanel can be change
        # -> has to be an integer
        H = int(matrix_shape[0] / (2 * n_maxpool))
        W = int(matrix_shape[1] / (2 * n_maxpool))
        self.input = int(out_chanel_2 * H * W)
        print(self.input)
        # Fully connected linear layer
        self.linear_1 = nn.Linear(self.input, 128)
        self.linear_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv_1(x))  # outputsize (H x W x 32)
        x = self.maxpool(x)  # outputsize (H/2 x W/2 x 32)
        x = torch.relu(self.conv_2(x))  # outputsize (H/2 x W/2 x 64)
        x = self.maxpool(x)  # outputsize (H/4 x W/4 x 64)
        # x = self.dropout(x) # Chance of inactivate a neuron
        x = x.view(-1, self.input)  # Flattern the data outputsize: (1 x 1 x self.input)
        x = torch.relu(self.linear_1(x))  # outputsize (128)
        x = torch.relu(self.linear_2(x))  # outputsize (10)
        return x


class linear_module_general(nn.Module):
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


class lenet5(nn.Module):
    """ LeNet-5 is a arcitecture shown to be good for MNIST digital numbers
    Consist of: 2 conv, 2 pooling, 3 linear
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvAlbin(nn.Module):
    def __init__(self, fully_connected_size, convolutional_size):
        super(ConvAlbin, self).__init__()
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