import numpy as np
from abc import ABC, abstractmethod
import extra_filer.extra_tools as tools
from sklearn.decomposition import PCA
import time
import os
import pandas as pd

tl = tools.EasyTimeLog()

class GeneralPCA(ABC):
    """
    Defines the structure and what methods a PCA calculating class must contain
    """
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data
        self.center_data()

    @abstractmethod
    def get_training_components(self, k):
        pass

    @abstractmethod
    def get_testing_components(self, k):
        pass

    def get_components(self, k):
        training_comp, training_time = self.get_training_components(k)
        testing_comp, testin_time = self.get_testing_components(k)
        return training_comp, training_time, testing_comp, testin_time

    def __len__(self):
        return len(self.testing_data)

    def center_data(self):
        mean = self.training_data.mean(axis=0)
        self.training_data = self.training_data - mean
        self.testing_data = self.testing_data - mean
        total_mean = self.training_data.mean()
        total_std = self.training_data.std()
        self.training_data = (self.training_data - total_mean)/total_std
        self.testing_data = (self.testing_data - total_mean)/total_std



class CovariancePCA(GeneralPCA):
    """
    Calculates PCA using SVD and a covariance matrix
    """
    def __init__(self, training_data, testing_data):
        tt = tl.start_log(f'training_size: {len(training_data)}')
        super().__init__(training_data, testing_data)
        C = np.cov(training_data, rowvar=False)
        self.V, _, _ = np.linalg.svd(C)
        tt.stop()
        self.initial_training_time = tt.time()

    def get_training_components(self, k):
        tt = tl.start_log(f'components: {k}, training_size: {len(self.training_data)}')
        comp = self.training_data @ self.V[:, :k]
        tt.stop()
        return comp, tt.time() + self.initial_training_time

    def get_testing_components(self, k):
        tt = tl.start_log(f'components: {k}, testing_size: {len(self.testing_data)}')
        comp = self.testing_data @ self.V[:, :k]
        tt.stop()
        return comp, tt.time()


class SvdPCA(GeneralPCA):
    """
    Calculates PCA using SVD on the original matrix
    """
    def __init__(self, training_data, testing_data):
        tt = tl.start_log(f'training_size: {len(training_data)}')
        super().__init__(training_data, testing_data)
        U, S, Vt = np.linalg.svd(self.training_data)
        self.U = U
        self.V = Vt.T
        self.S = S
        tt.stop()
        self.initial_training_time = tt.time()

    def get_training_components(self, k):
        tt = tl.start_log(f'components: {k}, training_size: {len(self.training_data)}')
        # Uk = self.U[:, :k]
        # Sk = np.diag(self.S[:k])
        comp = self.training_data @ self.V[:, :k]
        tt.stop()
        return comp, tt.time()+self.initial_training_time

    def get_testing_components(self, k):
        tt = tl.start_log(f'components: {k}, testing_size: {len(self.testing_data)}')
        comp = self.testing_data @ self.V[:, :k]
        tt.stop()
        return comp, tt.time()


class SklearnPCA(GeneralPCA):
    """
    Calculates PCA using the existing library sklearn
    """
    def __init__(self, training_data, testing_data):
        tt = tl.start_log(f'training_size: {len(training_data)}')
        super().__init__(training_data, testing_data)
        tt.stop()

    def get_training_components(self, k):
        tt = tl.start_log('TRAINING TIME', f'components: {k}, training_size: {len(self.training_data)}')
        pca = PCA(k)
        pca.fit(self.training_data)
        tt.stop()
        tt = tl.start_log(f'components: {k}, training_size: {len(self.training_data)}')
        comp = pca.transform(self.training_data)
        tt.stop()
        return comp, tt.time()

    def get_testing_components(self, k):
        tt = tl.start_log('TRAINING TIME', f'components: {k}, training_size: {len(self.training_data)}')
        pca = PCA(k)
        pca.fit(self.training_data)
        tt.stop()
        tt = tl.start_log(f'components: {k}, training_size: {len(self.training_data)}')
        comp = pca.transform(self.training_data)
        tt.stop()
        return comp, tt.time()

    def get_components(self, k):
        tt = tl.start_log('TRAINING TIME', f'components: {k}, training_size: {len(self.training_data)}')
        pca = PCA(k)
        pca.fit(self.training_data)
        comp_training = pca.transform(self.training_data)
        tt.stop()
        t1 = tt.time()

        tt = tl.start_log(f'testing testing', f'components: {k}, training_size: {len(self.training_data)}')
        comp_testing = pca.transform(self.testing_data)
        tt.stop()
        return comp_training, t1, comp_testing, tt.time()


class LoadPCA(GeneralPCA):
    """
    Laods a dataset in the same format as you'd get if you computed it
    """
    def __init__(self):
        """ Class for loading in previously calculated files returns 0 for the times """
        super().__init__(np.array([0]), np.array([0]))

    def get_training_components(self, k):
        train_array, _, _, _ = tools.load_pca_mnist(f'pca_transformed_MNIST_k={k}')
        return train_array, 0

    def get_testing_components(self, k):
        _, _, test_array, _ = tools.load_pca_mnist(f'pca_transformed_MNIST_k={k}')
        return test_array, 0

    def get_components(self, k):
        train_array, _, test_array, _ = tools.load_pca_mnist(f'pca_transformed_MNIST_k={k}')
        return train_array, 0, test_array, 0


class ReducedSizePCA(GeneralPCA):
    def __init__(self, pca: GeneralPCA, training_labels, reduction_factor=0.5):
        super().__init__(pca.training_data, pca.testing_data)
        self.pca = pca
        self.training_labels = training_labels
        self.new_training_labels = None
        self.reduction_factor = reduction_factor

    def get_training_components(self, k):
        tt = tl.start_log(f'Reducing size from {len(self.training_data)}', f'reduction factor={self.reduction_factor}', f'k={k}')
        training_components, training_time = self.pca.get_training_components(k)
        new_components, new_labels = tools.reduce_pca_dataset(training_components, self.training_labels, self.reduction_factor)
        self.new_training_labels = new_labels
        tt.stop()
        return new_components, tt.time() + training_time

    def get_testing_components(self, k):
        return self.pca.get_testing_components(k)

    def get_new_labels(self):
        return self.new_training_labels


def calculate_predictions(pca: GeneralPCA, labels_train, labels_test, k_list: list[int]):
    accuracies = np.zeros(len(k_list))
    training_times = np.zeros(len(k_list))
    testing_times = np.zeros(len(k_list))
    for i, k in enumerate(k_list):
        training_scores, training_time, testing_scores, testing_preprocess_time = pca.get_components(k)
        if isinstance(pca, ReducedSizePCA):
            labels_train = pca.get_new_labels()  # if it is a reduced size pca then the labels need to be refreshed
        print(f'k: {k}, len(training): {len(labels_train)}')
        t0 = time.time()
        for j, row in enumerate(testing_scores):
            if j % 200 == 0:
                print(j)
            distances = np.sum(np.square(training_scores-row), axis=1)
            predicted_number = labels_train[np.argmin(distances)]
            if predicted_number == labels_test[j]:
                accuracies[i] += 1
        testing_times[i] = time.time() - t0 + testing_preprocess_time
        training_times[i] = training_time
    accuracies /= len(pca)
    testing_times /= len(pca)
    return accuracies, training_times, testing_times


def test_so_that_all_pca_methods_give_the_same_resutl():
    np.random.seed(10)
    training = np.random.rand(10, 6)
    np.random.seed(12)
    testing = np.random.rand(1, 6)
    pca1 = CovariancePCA(training, testing)
    pca2 = SvdPCA(training, testing)
    pca3 = SklearnPCA(training, testing)
    a1, _, a2, _ = pca1.get_components(2)
    b1, _, b2, _ = pca2.get_components(2)
    c1, _, c2, _ = pca3.get_components(2)
    assert np.allclose(a1, b1)
    assert np.allclose(a1, c1)
    assert np.allclose(a2, b2)
    assert np.allclose(a2, c2)
    # This does not always compute since the sign of the eigenvectors are arbitrary


def test_pca(save=True, training_set_sizes=None, k_list=None):
    # training_set_sizes = [-1, -2, -3, 1000, 5000, 10000, 30000, 60000]
    if training_set_sizes is None:
        training_set_sizes = [999, 1000, 1001, 5000, 10000]
    if k_list is None:
        k_list = [1, 2, 3, 5, 8, 14, 20, 28, 64, 128]

    accuracies_list = []
    training_times_list = []
    testing_times_list = []
    test_data, test_labels = tools.load_mnist(False)
    test_data = np.array(test_data )
    test_labels = np.array(test_labels )
    for i, size in enumerate(training_set_sizes):
        if size == -1:
            train_data, train_labels = tools.load_mnist(True)
            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            pca = ReducedSizePCA(CovariancePCA(train_data, test_data), train_labels, 0.5)
        elif size == -2:
            train_data, train_labels = tools.load_mnist(True)
            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            pca = ReducedSizePCA(CovariancePCA(train_data, test_data), train_labels, 0.75)
        elif size == -3:
            train_data, train_labels = tools.load_mnist(True)
            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            pca = ReducedSizePCA(CovariancePCA(train_data, test_data), train_labels, 1)
        else:
            train_data, train_labels = tools.load_mnist(True, size)
            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            pca = CovariancePCA(train_data, test_data)
        accuracy, train_time, test_time = calculate_predictions(pca, train_labels, test_labels, k_list)
        accuracies_list.append(accuracy)
        training_times_list.append(train_time)
        testing_times_list.append(test_time)

    if save:
        save_dir = os.path.join(tools.get_project_root(), 'pca_image_classification', 'results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pd.DataFrame(
            accuracies_list, columns=k_list, index=np.array(training_set_sizes)).to_csv(
                os.path.join(save_dir, 'accuracy.csv'), sep=';')

        pd.DataFrame(
            training_times_list, columns=k_list, index=np.array(training_set_sizes)).to_csv(
                os.path.join(save_dir, 'training_time.csv'), sep=';')

        pd.DataFrame(
            testing_times_list, columns=k_list, index=np.array(training_set_sizes)).to_csv(
                os.path.join(save_dir, 'testing_time.csv'), sep=';')


if __name__ == '__main__':
    test_pca(True)

