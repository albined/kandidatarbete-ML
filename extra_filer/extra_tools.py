import datetime
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from warnings import warn
# from cupyx.scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist
from torchvision import datasets
import time
import inspect


class EasyTimeLog:
    """
    Creates an easy way to log the time it takes for different functions and such to execute
    no parameters

    Use log(time, model, etc)

    Use start_log('model' ... etc) to start  a new log.
    The start_log returns a TimeTracker object
    call timetracker.stop() to stop the time tracker
    This logs the time it took for the function to execute in a tidslog.txt file in the folder extra_filer
    """

    def __init__(self):
        self.path = os.path.join(get_project_root(), 'extra_filer', 'tidslog.txt')
        if not os.path.exists(self.path):
            # Create file if it does not exist
            with open(self.path, 'a'):
                pass

    def start_log(self, *args):
        other = [str(arg) for arg in args]
        caller = inspect.stack()[1][3]
        s = '    '.join([f'Date: {datetime.datetime.now()}', f'From: {caller}', *other, 'Time: {}']) + '\n'
        return TimeTracker(s, self.path)

    def log(self, log_time=None, *args):
        other = [str(arg) for arg in args]
        caller = inspect.stack()[1][3]
        s = '    '.join([f'Date: {datetime.datetime.now()}', f'From: {caller}', *other, f'Time: {log_time}']) + '\n'
        with open(self.path, 'a') as f:
            f.write(s)



class TimeTracker:
    def __init__(self, log_string, path):
        self.start_time = time.time()
        self.log_string = log_string
        self.path = path

    def stop(self):
        """ Stops current time and logs the time the activity took """
        with open(self.path, 'a') as f:
            f.write(self.log_string.format(time.time() - self.start_time))

    def time(self):
        """ Gets the time since the object was created """
        return time.time() - self.start_time





def get_project_root():
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    project_root = os.path.dirname(current_directory)
    return project_root


def append_row_to_csv(array_to_append, save_path, relative_root=False):
    warn('This is deprecated, use append_matrix_to_csv', DeprecationWarning, stacklevel=2)
    if relative_root:
        save_path = os.path.join(get_project_root(), save_path)

    if not os.path.exists(save_path):
        # Create file if it does not exist
        with open(save_path, 'a'):
            pass

    df = pd.DataFrame(array_to_append.reshape(1, len(array_to_append)))
    df.to_csv(save_path, index=False, header=False, mode='a', sep=';')
    print(f'Saved file to {save_path}')


def append_matrix_to_csv(array_to_append, save_path, relative_root=False, folder=None):
    if folder is not None:
        if relative_root:
            folder_path = os.path.join(get_project_root(), folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    if relative_root:
        save_path = os.path.join(get_project_root(), folder, save_path)

    if not os.path.exists(save_path):
        # Create file if it does not exist
        with open(save_path, 'a'):
            pass

    df = pd.DataFrame(array_to_append)
    df.to_csv(save_path, index=False, header=False, mode='a', sep=';')
    print(f'Saved file to {save_path}')


def load_mnist(training=True, max_number=-1, keep_square=False, normalize=True, dtype='float32'):
    """
    Loads the MNIST dataset as NumPy arrays.
    args:
        training (bool): Whether to load the training data (True) or the test data (False).
        max_number (int): The maximum digit label to include in the dataset (default is 9).
        keep_square (bool): Whether to reshape the data to n x 784 (False) or keep it at n x 28 x 28 (True)
    returns:A tuple containing the data and labels as NumPy arrays.
    """
    save_path = os.path.join(get_project_root(), 'MNIST')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the MNIST dataset as NumPy arrays
    data_training = datasets.MNIST(root=save_path, train=True, download=True).train_data.numpy()
    labels_training = datasets.MNIST(root=save_path, train=True, download=True).train_labels.numpy()
    data_testing = datasets.MNIST(root=save_path, train=False, download=True).test_data.numpy()
    labels_testing = datasets.MNIST(root=save_path, train=False, download=True).test_labels.numpy()

    if normalize:
        data_training = data_training/255
        data_testing = data_testing/255
        training_mean = 0.1306604762738429  # data_training.mean()
        training_std = 0.3081078038564622  # data_training.std()
        data_training = (data_training - training_mean)/training_std
        data_testing = (data_testing - training_mean)/training_std


    if not keep_square:
        data_training = data_training.reshape(-1, 28 ** 2)
        data_testing = data_testing.reshape(-1, 28 ** 2)


    if training:
        data = data_training
        labels = labels_training
    else:
        data = data_testing
        labels = labels_testing

    if not keep_square:
        data = data.reshape(-1, 28 ** 2)

    if max_number > 0:
        data = data[:max_number]
        labels = labels[:max_number]

    data = data.astype(dtype)

    return data, labels


def load_pca_mnist(directory_name, create_if_nonexistent=True, dtype='float32'):
    """
    Easy way to load the PCA modified dataset.
    :param directory_name: Directory name inside "Generated dataset variants" ex "pca_transformed_MNIST_k=10"
    :return: training set, training labels, testing set, testing labels
    """
    k = None
    if k is None and 'k=' in directory_name:
        k = int(directory_name.split('k=')[-1])

    load_path = os.path.join(get_project_root(), 'Generated dataset variants', directory_name)
    if not os.path.exists(load_path):
        if create_if_nonexistent:
            from extra_filer.dataset_variant_generator import pca_library_main
            pca_library_main(k)
        else:
            raise Exception(f"The specified directory does not exist at: {load_path}")

    files = glob.glob(os.path.join(load_path, '*'))
    test_label_path = None
    train_label_path = None
    test_path = None
    train_path = None
    for file_path in files:
        label = False
        test = False
        if 'label' in file_path:
            label = True
        if 'test' in file_path:
            test = True
        if label and test:
            test_label_path = file_path
        elif label:
            train_label_path = file_path
        elif test:
            test_path = file_path
        else:
            train_path = file_path
    if test_path is None or train_path is None or test_label_path is None or train_label_path is None:
        raise Exception(f"The specified directory does not contain all necessary files: {load_path}")
    test_label_array = pd.read_csv(test_label_path, sep=';', header=None).to_numpy()
    train_label_array = pd.read_csv(train_label_path, sep=';', header=None).to_numpy()
    test_array = pd.read_csv(test_path, sep=';', header=None).to_numpy()
    train_array = pd.read_csv(train_path, sep=';', header=None).to_numpy()

    train_array = train_array.astype(dtype)
    test_array = test_array.astype(dtype)

    return train_array, train_label_array, test_array, test_label_array


def load_svd_mnist(directory_name, reshape=False, k=None, create_if_nonexistent=True, dtype='float32'):
    """
    Easy way to load the modified V or U datasets. Can return them both as 2d matrix of vectors or as a flattened version
    :param create_if_nonexistent: creates the dataset if it doesn't exist
    :param directory_name: Directory of the folder inside "Generated dataset variants"
    :param reshape: Reshapes each row into 2d matrixes for each eigenvector
    :param k: the truncation number
    :return: training set, training labels, testing set, testing labels
    """
    if k is None and 'k=' in directory_name:
        k = int(directory_name.split('k=')[-1])
    elif reshape:
        raise Exception(
            f'The directory either has to contain the number of principal components k, or it has to be specified if reshape is true')

    load_path = os.path.join(get_project_root(), 'Generated dataset variants', directory_name)
    if not os.path.exists(load_path):
        if create_if_nonexistent:
            from extra_filer.dataset_variant_generator import svd_libraries_main
            svd_libraries_main(k)
        else:
            raise Exception(f"The specified directory does not exist at: {load_path}")
    files = glob.glob(os.path.join(load_path, '*'))
    test_label_path = None
    train_label_path = None
    test_path = None
    train_path = None
    for file_path in files:
        label = False
        test = False
        if 'label' in file_path:
            label = True
        if 'test' in file_path:
            test = True
        if label and test:
            test_label_path = file_path
        elif label:
            train_label_path = file_path
        elif test:
            test_path = file_path
        else:
            train_path = file_path
    if test_path is None or train_path is None or test_label_path is None or train_label_path is None:
        raise Exception(f"The specified directory does not contain all necessary files: {load_path}")
    test_label_array = pd.read_csv(test_label_path, sep=';', header=None).to_numpy()
    train_label_array = pd.read_csv(train_label_path, sep=';', header=None).to_numpy()
    test_array = pd.read_csv(test_path, sep=';', header=None).to_numpy()
    train_array = pd.read_csv(train_path, sep=';', header=None).to_numpy()

    if reshape:
        test_array = test_array.reshape((len(test_array), -1, k))
        train_array = train_array.reshape((len(test_array), -1, k))

    train_array = train_array.astype(dtype)
    test_array = test_array.astype(dtype)

    return train_array, train_label_array, test_array, test_label_array


def reduce_pca_dataset(dtrain, train_labels, threshold_mult=0.5, dtype='float32'):
    """
    Combines similar images in the training set.
    If the
    threshold_mult: If the distance between two images is smaller than distance_mult*average_distance then they will be
                    combined
    """
    dtrain=dtrain.get()
    train_labels = train_labels.get()
    dtrain_reduced = []
    dlabels_reduced = []
    for n in range(10):
        dnum_new = []
        dnum = dtrain[(train_labels == n).reshape([-1])]
        # Compute the average distance for the first 10 occurences of each number since
        # that should be close enough
        average_distance = sum([np.sum(np.square(dnum - dnum[i]), axis=1).mean() for i in range(10)]) / 10
        threshold = average_distance * (threshold_mult**2)
        distances = np.array(cdist(dnum, dnum, metric='sqeuclidean'))
        pairs = np.argwhere(distances < threshold)
        pairs = [(i, j) for (i, j) in pairs if i < j]  # remove duplicates and self-matches
        pair_dict = defaultdict(set)
        used_set = set()
        for idx, (i, j) in enumerate(pairs):
            if idx % 1000 == 0:
                pass
                # print(f'idx: {idx} out of: {len(pairs)}')
            if not i in used_set:
                used_set.add(j)
                pair_dict[i].add(i)
                pair_dict[i].add(j)
        for key, item in pair_dict.items():
            dnum_new.append(dnum[list(item)].mean(axis=0))
        dtrain_reduced += dnum_new
        dlabels_reduced += [n]*len(dnum_new)

    dtrain_reduced = np.array(dtrain_reduced)
    dlabels_reduced = np.array(dlabels_reduced)

    dtrain_reduced = dtrain_reduced.astype(dtype)

    return dtrain_reduced, dlabels_reduced


