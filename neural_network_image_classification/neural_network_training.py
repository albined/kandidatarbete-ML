import os.path
import warnings

import cupy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
# import extra_filer.extra_tools as tools
from neural_network_models import *
import extra_filer.extra_tools as tools
from neural_network_image_classification.neural_network_models import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import torch.nn.functional as F

timelog = tools.EasyTimeLog()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleDataset(Dataset):
    """A class represent data, Are needed to make a dataloader to interate through when training and testing the model'
    The "SimpleDataset" has __len__ and __getitem__ as abstractfunction -> therefore this classneed to have the same
    """

    def __init__(self, data, labels):
        self.data = data
        self.targets = labels
        if len(data) != len(labels):
            warnings.warn('The labels and the images are not the same length')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.targets[idx]
        return torch.tensor(data, device=device), torch.tensor(int(label), device=device)


class ScoreDataset(Dataset):
    """ This is a dataset for loading the pca dimensionality reduced image """

    def __init__(self, k, datatype='train', dataset_name='MNIST'):
        train_array, train_label_array, test_array, test_label_array = tools.load_pca_mnist(
            f'pca_transformed_{dataset_name}_k={k}')
        if datatype == 'train':
            print(f'Loading training set pca_transformed_{dataset_name}_k={k}')
            self.data = torch.tensor(train_array, device=device)
            self.label = torch.tensor(train_label_array, device=device)
        else:
            print(f'Loading testing set pca_transformed_{dataset_name}_k={k}')
            self.data = torch.tensor(test_array, device=device)
            self.label = torch.tensor(test_label_array, device=device)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label


class SVDDataset(Dataset):
    """ Parent for Uk and Vk dataset"""

    def __init__(self, k, u_v, datatype='train', reshape=True, dataset_name='MNIST'):
        train_array, train_label_array, test_array, test_label_array = tools.load_svd_mnist(
            f'{u_v}_svd_transformed_{dataset_name}_k={k}', reshape=reshape)
        if datatype == 'train':
            print(f'Loading training set pca_transformed_{dataset_name}_k={k}')
            self.data = torch.tensor(train_array, device=device)
            self.label = torch.tensor(train_label_array, device=device)
        else:
            print(f'Loading testing set pca_transformed_{dataset_name}_k={k}')
            self.data = torch.tensor(test_array, device=device)
            self.label = torch.tensor(test_label_array, device=device)
        self.reshape = reshape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.reshape:
            data = data[None, :, :]
        label = self.label[idx]
        return data, label


class UkDataset(SVDDataset):
    """ Dataset for loading in the Uk dataset """

    def __init__(self, k, datatype='train', reshape=True, dataset_name='MNIST'):
        super().__init__(k, 'U', datatype=datatype, reshape=reshape, dataset_name=dataset_name)


class VkDataset(SVDDataset):
    """ Dataset for loading in the Vk dataset """

    def __init__(self, k, datatype='train', reshape=True, dataset_name='MNIST'):
        super().__init__(k, 'V', datatype=datatype, reshape=reshape, dataset_name=dataset_name)


class NormalMNISTDataset(Dataset):
    """ Normal dataset just to compare the plots """

    def __init__(self, datatype='train', square=False):
        if datatype == 'train':
            data, label = tools.load_mnist()
        else:
            data, label = tools.load_mnist(False)
        if square:
            data = data.reshape([-1, 1, 28, 28])
        self.data = torch.tensor(data, device=device)
        self.label = torch.tensor(label, device=device)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label


def get_MNIST():
    """A methods which will return the MNIST data in a form of train and test data as tensors"""
    # Defines how the data would look like, For example somtimes it flips so the data are up side down, to learn better. OBS transform is only a parameter
    # Change this later
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # resize the input image to (256, 256)
        transforms.RandomCrop(224),  # randomly crop the image to (224, 224)
        transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
    ])
    mnist_train = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    return mnist_train, mnist_test


def get_ISIC():
    """ A methods which will return the ISIC data in a form of train and test data as tensors"""
    # Maybe use svd_k on ISIC database if it's working to train on U_k, Z_k and V_k
    pass


def truncated_svd(A, k: int, SVD_k=False):
    """ A method for truncated svd, which return a the svd, U, Z and V truncated with rang k
    Param: A = data, k = Rang, SVD_k = if you want the svd_k
    Return: U, Z, V and mabe svd all trunkated with rang k
    """
    U, Z, V = np.linalg.svd(A, full_matrices=True)  # Note: Z is not a diagonal matrix, only a vector
    # Checks the dimension on the input and then Calculate truncated U, Z and V
    if A.ndim == 3:
        U_k = U[:, :, :k]  # shape (60000,28,k)
        Z_k = Z[:, :k]  # shape (60000,k)
        V_k = V[:, :k, :]  # shape (60000,k,28)
        if SVD_k is True:
            # Calculate the truncated SVD
            # must use np.einsum because np.array matrix has a limit on the shape, np.einsum will multiply elementwise in shape of 'ijk,ik,ikl->ijl'
            svd_k = np.einsum('ijk,ik,ikl->ijl', U_k, Z_k, V_k)  # shape (60000,28,28)
            return U_k, Z_k, V_k, svd_k
    elif A.ndim == 2:
        U_k = U[:, :k]  # shape (28,k)
        Z_k = Z[:k]  # shape (k)
        V_k = V[:k, :]  # shape (k,28)
        if SVD_k is True:
            # Calculate the truncated SVD
            svd_k = np.matrix(U_k) * np.diag(Z_k) * np.matrix(V_k)  # shape (28,28)
            return U_k, Z_k, V_k, svd_k
    else:
        ValueError("Input matrix should be 2D or 3D.")
    return U_k, Z_k, V_k


def pca_matrix(A, k):
    """ Calculates the score matrix by truncated svd and given rang
    param: A = matrix, k = rang
    return: score matris with rang k as a tensor
    """
    A_mean = torch.mean(A.float(), dim=2, keepdim=True)
    A = A - A_mean
    U_k, Z_k, _ = truncated_svd(A, k, SVD_k=False)
    # Calculation the score matix
    T_k_nparray = np.einsum('ijk,ik->ijk', U_k, Z_k)  # np.einsum will multiply elementwise in shape of 'ijk,ik->ijk'
    T_k_tensor = torch.from_numpy(T_k_nparray)
    T_k_tensor = T_k_tensor.unsqueeze(1)
    return T_k_tensor


def load_data(path_name: str):
    """ Function that loads csv files
    Input: The path to the data as a string
    Output: Data as
    """
    data = pd.read_csv(path_name, header=None, sep=';', dtype=float)
    matrix = data.values
    return matrix


def plot_images(data):
    """ A method that's plots the given matrix (image)
    Param: data = matrix (image)
    """
    fig = plt.figure
    plt.imshow(data, cmap='gray')
    plt.show()


def train_module(model: nn.modules, train_data: DataLoader, name: str, epochs: int):
    """ A method which trains a module
     Param: module = NN model, train_data = The trainset, name = the name of model that will be saved
     Return: The total time for the training
       """
    # optimizer = optim.SGD(module.parameters(), lr=0.01)  # Stochastic Gradient Decent
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss = nn.CrossEntropyLoss()
    time_1 = time.time()  # start time
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for (x, y) in train_data:
            y = torch.flatten(y)
            optimizer.zero_grad(set_to_none=True)  # resets the gradients before countinue to the next batch
            prediction = model(x)  # predict and makes the forward
            # luss = loss(prediction, y)  # Calculate the loss using Crossentropy
            luss = F.nll_loss(prediction, y)
            luss.backward()  # compute the gradients
            optimizer.step()  # updates the weights

    time_2 = time.time()  # end time
    train_time = time_2 - time_1  # total time

    # Saves the model (assume you are in the big folder)
    path = os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'saved_models', name + '.pth')
    torch.save(model.state_dict(), path)

    return train_time  # the model don't need to be returned, it's already updated


def load_model(model: nn.Module, name: str):
    """ A method which loads the saved parameters into a model.
     Param: module = NN model, name = the name of model the saved model
       """
    # loads the model (assume you are in the big folder)
    path = os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'saved_models', name + '.pth')
    model.load_state_dict(torch.load(path))
    model.to(device)


def train_score_nn(k, epochs=10, hidden_layer_size=128):
    """
    Fast way to train a score-matrix dataset
    Trains a Neural network with only linear layers
    """
    dataset = ScoreDataset(k)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    module = linear_module_general(k, hidden_layer_size)
    module.to(device)
    training_time = train_module(module, dataloader,
                                 f'linear_model_score_epoch_{epochs}_k={k}_layersize_{hidden_layer_size}',
                                 epochs=epochs)
    timelog.log(training_time, f'k={k}', f'hidden layer size {hidden_layer_size}', f'length dataset: {len(dataset)}',
                f'epochs={epochs}')
    return training_time


def test_score_nn(k, epochs=10, hidden_layer_size=128, tests=1):
    module = linear_module_general(k, hidden_layer_size)
    module.to(device)
    load_model(module, f'linear_model_score_epoch_{epochs}_k={k}_layersize_{hidden_layer_size}')
    dataset = ScoreDataset(k, datatype='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    result_array = np.array([test_model(module, dataloader) for _ in range(tests)])
    accuracy, testing_time = result_array.mean(axis=0)
    testing_time /= len(dataset)
    timelog.log(testing_time, f'k={k}', f'hidden layer size {hidden_layer_size}', f'length dataset: {len(dataset)}',
                f'epochs={epochs}', f'accuracy {accuracy}')
    return accuracy, testing_time


def train_svd_nn(k, u_v, epochs=10, hidden_layer_size=128):
    """ Trains the U or V dataset on a linear neural network """
    dataset = SVDDataset(k, u_v, reshape=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    module = linear_module_general(k * 28, hidden_layer_size)  # We have k*28 input values
    module.to(device)
    training_time = train_module(module, dataloader,
                                 f'linear_model_{u_v}_epochs={epochs}_k={k}_layersize={hidden_layer_size}',
                                 epochs=epochs)
    timelog.log(training_time, f'k={k}', f'type: {u_v}', f'hidden layer size {hidden_layer_size}',
                f'length dataset: {len(dataset)}', f'epochs={epochs}')
    return training_time


def test_svd_nn(k, u_v, epochs=10, hidden_layer_size=128, tests=1):
    module = linear_module_general(k * 28, hidden_layer_size)
    module.to(device)
    load_model(module, f'linear_model_{u_v}_epochs={epochs}_k={k}_layersize={hidden_layer_size}')
    dataset = SVDDataset(k, u_v, reshape=False, datatype='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    result_array = np.array([test_model(module, dataloader) for _ in range(tests)])
    accuracy, testing_time = result_array.mean(axis=0)
    testing_time /= len(dataset)
    timelog.log(testing_time, f'k={k}', f'type: {u_v}', f'hidden layer size {hidden_layer_size}',
                f'length dataset: {len(dataset)}', f'epochs={epochs}', f'accuracy {accuracy}')
    return accuracy, testing_time


def train_normal_mnist_nn(epochs=10, hidden_layer_size=128):
    """ Trains the U or V dataset on a linear neural network """
    torch.no_grad()
    dataset = NormalMNISTDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    module = linear_module_general(784, hidden_layer_size)  # We have k*28 input values
    module.to(device)
    training_time = train_module(module, dataloader,
                                 f'normal_mnist_linear_epochs={epochs}_layersize={hidden_layer_size}', epochs=epochs)
    timelog.log(training_time, f'hidden layer size {hidden_layer_size}', f'length dataset: {len(dataset)}',
                f'epochs={epochs}')
    return training_time


def test_normal_mnist_nn(epochs=10, hidden_layer_size=128, tests=1):
    module = linear_module_general(784, hidden_layer_size)
    module.to(device)
    load_model(module, f'normal_mnist_linear_epochs={epochs}_layersize={hidden_layer_size}')
    dataset = NormalMNISTDataset(datatype='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    result_array = np.array([test_model(module, dataloader) for _ in range(tests)])
    accuracy, testing_time = result_array.mean(axis=0)
    testing_time /= len(dataset)
    timelog.log(testing_time, f'hidden layer size {hidden_layer_size}', f'length dataset: {len(dataset)}',
                f'epochs={epochs}', f'accuracy {accuracy}')
    return accuracy, testing_time


def train_cnn_mnist_nn(epochs=10, linear_layer_size=128, convolutional_layer_size=16):
    """ Trains the U or V dataset on a linear neural network """
    torch.no_grad()
    dataset = NormalMNISTDataset(square=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    module = ConvAlbin(linear_layer_size, convolutional_layer_size)
    module.to(device)
    training_time = train_module(module, dataloader,
                                 f'cnn_mnist_epochs={epochs}_layersize={linear_layer_size}_cnn{convolutional_layer_size}',
                                 epochs=epochs)
    timelog.log(training_time, f'hidden layer size {linear_layer_size}',
                f'length dataset: {len(dataset)} and {convolutional_layer_size}', f'epochs={epochs}')
    return training_time


def test_cnn_mnist_nn(epochs=10, linear_layer_size=128, convolutional_layer_size=16, tests=1):
    module = ConvAlbin(linear_layer_size, convolutional_layer_size)
    module.to(device)
    load_model(module, f'cnn_mnist_epochs={epochs}_layersize={linear_layer_size}_cnn{convolutional_layer_size}')
    dataset = NormalMNISTDataset(datatype='test', square=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    result_array = np.array([test_model(module, dataloader) for _ in range(tests)])
    accuracy, testing_time = result_array.mean(axis=0)
    testing_time /= len(dataset)
    timelog.log(testing_time, f'hidden layer size {linear_layer_size} and {convolutional_layer_size}',
                f'length dataset: {len(dataset)}', f'epochs={epochs}', f'accuracy {accuracy}')
    return accuracy, testing_time


def test_model(model: nn.Module, test_data: DataLoader):
    """ Testing the model and calculate the prediction rate
    param: model = the model to test, test_data = the test data
    return: prediction rate
    """
    model = model.eval()  # sets the model in eveulation mode, because we dont want to update any weights
    correct = 0
    total_matrixes = 0

    t0 = time.time()
    for (data, labels) in test_data:
        labels = labels.flatten()
        outputs = model(data)
        # Want to calculate max of each row inthe matrix, there for 1, (0 for colum)
        # The output will be a tuple (value, index) but we are only intrested in the index because it can be compered to the labels
        _, predicted = torch.max(outputs.data, 1)

        # Calculate number of correct predictions
        # (predicted == labels) -> Comperes the predicted valus with the labels and returns a vector with boolens
        # .sum() -> sums the True boolen to a tensor
        # .item() -> Transforms the tensor to an integer.
        correct += (predicted == labels).sum().item()

        # calculate the total number of images/matrixes in a batch
        # Shape (x,y,z) -> 0 corresponds to batchsize
        total_matrixes += labels.size(0)
    time_taken = time.time() - t0

    prediction_rate = correct / total_matrixes

    return prediction_rate, time_taken


def get_project_root():
    """ Taken from "Tools" (can't find the folder)
    """
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    project_root = os.path.dirname(current_directory)
    return project_root


def test_pca_nn(k, epochs: int, hidden_layer_size: int, train: torch.Tensor, test: torch.Tensor):
    name = 'linear_model_pca,k=' + str(k) + ',layer=' + str(hidden_layer_size)
    # formating data
    T_k_train = pca_matrix(train.data, k)
    T_k_test = pca_matrix(test.data, k)
    train_shape = T_k_train.shape[2] * T_k_train.shape[3]
    T_k_train = SimpleDataset(T_k_train, train.targets)
    T_k_test = SimpleDataset(T_k_test, test.targets)
    # Makes the Dataloader
    data_train = DataLoader(T_k_train, batch_size=64, shuffle=True)
    data_test = DataLoader(T_k_test, batch_size=64, shuffle=True)
    # defines the nn
    lnn = linear_module_general(train_shape, hidden_layer_size)
    lnn.to(device)
    # train
    train_time = train_module(lnn, data_train, name, epochs)
    # test
    pre_rate, pre_time = test_model(lnn, data_test)
    pre_time = pre_time / len(T_k_test)

    return pre_rate, train_time, pre_time


def test_LeNet5(train_data, test_data, epochs=10):
    """ A function thats train and test the LeNet5 nn
    NOTE: Only works for normal image (not fomated)
    """
    # Loads the model
    LeNet5 = lenet5()
    LeNet5_name = 'LeNet5_normal'
    # Train
    train_time = train_module(model=LeNet5, train_data=train_data, name=LeNet5_name, epochs=epochs)
    # Test
    pre_rate, pre_time = test_model(model=LeNet5, test_data=test_data)

    return pre_rate, train_time, pre_time


if __name__ == '__main__':
    train, test = get_MNIST()  # .data = images     .targets = labels
    # transform to special class called DataLoader,which pytorch are traing models with.
    data_train = DataLoader(train, batch_size=64, shuffle=True)
    data_test = DataLoader(test, batch_size=64, shuffle=True)

    # NOTE (for linear_module())
    # Epoch = 10, took 504.79717659950256 seconds to train, 0.9346 % predicting rate
    # Epoch = 15, took 708.9179368019104 seconds to train, 0.8564 % predicting rate  -> OVERFITTING

    # NOTE (for conv_module())
    # Epoch = 5, took 718.831481218338 seconds to train, 0.9644 % predicting rate

    # NOTE (for conv_module_general())
    # Epoch = 5, k = 6,  took 136.98681330680847 seconds to train, 0.098% or 0.8736% predicting rate
    # Epoch = 10, k = 6,  took 299.85329127311707 seconds to train, 0.8983 % predicting rate

    # NOTE: Testing the code (only for myself)
    """ Data processing of pcamatrix"""
    # k = 6
    # T_k_train = score_matrix(train.data,k)
    # T_k_test = score_matrix(test.data,k)
    # train_shape = T_k_train.shape[1:] # Calculate the input shape of matrix to NN
    # Addning a chanel size so it will be at same format as original data. (not nicly done)
    # T_k_train = T_k_train.unsqueeze(1)
    # T_k_test = T_k_test.unsqueeze(1)
    # Makes a class inheritate from SimpleDataset which contains the prossessed data and it's labels
    # T_k_train = SimpleDataset(T_k_train, train.targets)
    # T_k_test = SimpleDataset(T_k_test, test.targets)
    # Makes the Dataloader
    # data_train = DataLoader(T_k_train, batch_size=64, shuffle=True)
    # data_test = DataLoader(T_k_test, batch_size=64, shuffle=True)

    """ If you want to try to train a model"""
    # model = linear_module_general()
    # model_name = 'linear_model'
    # train_time = train_module(model=model, train_data=data_train, name=model_name, epochs=10)
    # pre_rate, _ = test_model(model=model, test_data=data_test)
    # print('prediction rate: ',pre_rate)
    # print('training time',train_time)

    """ If you want to load a model"""
    # load_model(cnn,cnn_name)
    # pre_rate, _ = test_model(cnn, data_test)
    # print('prediction rate: ',pre_rate)

    """ Calculate Svd_k and score matrix"""
    # k = 17
    # U_k, Z_k, V_k,svd_k= truncated_svd(train.data[1], k, SVD_k=True)
    # print(Z_k)
    # T_k = pca_matrix(train.data,k)
    # train_shape = T_k.shape[1:]









