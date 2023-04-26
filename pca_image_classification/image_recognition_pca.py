import warnings
from matplotlib import pyplot as plt
from functools import wraps, cache
from extra_filer.extra_tools import *
from warnings import warn
import numpy as np
import math
import os

import matplotlib
matplotlib.use('tkagg')

# Av någon anledning får jag ett error utan denna. Antagligen något library som bråkar.
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap


@timing
@cache
def import_mnist(bool_training_data: bool, number_read: int = -1, separate_testing_data=True, max_training_number=None):
    warn('This is deprecated, use load_mnist from extra_tools instead', DeprecationWarning, stacklevel=2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if bool_training_data or not separate_testing_data:
        data = pd.read_csv(os.path.join(script_dir, 'mnist_train.csv'), delimiter=',', header=None, dtype='int64').to_numpy()
    else:
        data = pd.read_csv(os.path.join(script_dir, 'mnist_test.csv'), delimiter=',', header=None, dtype='int64').to_numpy()

    pixel_array = None
    labels = None
    if separate_testing_data:
        if number_read <= -1:
            pixel_array = data[:, 1:]
            labels = data[:, 0]
        else:
            pixel_array = data[:number_read, 1:]
            labels = data[:number_read, 0]
    else:
        # If separate_testing_data = False => it reads both the training data and testing data from the same file
        # The training data will be the first
        if number_read <= 0:
            warnings.warn('If the testing and training data are not separate you have to specify a \'number_read\' > 0')
        if max_training_number is not None and number_read + max_training_number > data.shape[1]:
            warnings.warn(f'Max training number is larger than it can be with the current number_read: \n'
                          f'Current: {max_training_number}, max: {data.shape[1]-number_read}')
        else:
            if bool_training_data:
                labels = data[:-number_read, 0]
                pixel_array = data[:-number_read, 1:]if max_training_number is None else data[:max_training_number, 1:]
            else:
                labels = data[-number_read:, 0]
                pixel_array = data[-number_read:, 1:]

    print(f'Imported {len(labels)} handwritten numbers')
    return pixel_array, labels


@timing
def calculate_predictions(training_data, testing_data, dim_to_keep, N_test):
    """
    This one is the same as the one Larisa used in matlab
    Note that this one computes the svd for the covariance matrix.
    Because of this the singular values etc are NOT! the same as if they were computed for
    X. However the Right singular vectors are the same!!!!!
    :param training_data:
    :param testing_data:
    :param dim_to_keep:
    :param N_test:
    :return:
    """

    # Center the data
    training_mean = training_data.mean(axis=0)
    testing_mean = testing_data.mean(axis=0)
    pixels_train_adj = training_data - training_mean
    pixels_test_adj = testing_data - training_mean
    # I matlab skriptet som jag skrev om var det sådär men borde man inte centrera test_datan med test_mean?

    # Calculate the covariance matrix and SVD
    # In python np.linalg.svd returns U, the diagonal of S and V^H
    C = np.cov(pixels_train_adj, rowvar=False)
    U, S_diag, Vh = np.linalg.svd(C)
    S = np.diag(S_diag)
    V = Vh.T

    # C = V L V^T och V är samma som i SVD av X = U S V^T.
    # Kanske lite förvirrande att använda SVD av Covariance därför? I allafall med namnen som de är

    # Ett annat sätt som borde fungera är detta med economy-size svd:




    PCA_test_list = []
    y_pred_data = np.zeros([len(dim_to_keep), len(labels_test)])
    pred_rate = np.zeros(len(dim_to_keep))
    falses = np.zeros([len(N_dim_to_keep), N_test])

    time_dict = {}

    for i in range(len(dim_to_keep)):
        t0 = time.time()
        k = dim_to_keep[i]
        PCA = np.matmul(pixels_train_adj, V[:, :k])
        PCA_test = np.matmul(pixels_test_adj, V[:, :k])
        PCA_test_list.append(PCA_test)

        n_correct = 0
        for j in range(len(labels_test)):
            print(f'Calculating: {i*len(labels_test) + j}/{len(dim_to_keep)*len(labels_test)}')
            diff_matrix = PCA - PCA_test[j, :]
            norm_matrix = np.linalg.norm(diff_matrix, axis=1)
            most_similar_index = np.where(norm_matrix == norm_matrix.min())[0][0]
            predicted_number = labels_train[most_similar_index]
            if predicted_number == labels_test[j]:
                n_correct += 1
            else:
                falses[i, predicted_number - 1] += 1

            y_pred_data[i, j] = predicted_number
        pred_rate[i] = n_correct / len(labels_test)
        time_dict[k] = (time.time() - t0)/N_test

    # Since U and S are not the U and S for Xs SVD I will replace them with zeros here so we won't get confused.
    return PCA_test_list, y_pred_data, pred_rate, np.zeros(U.shape), np.zeros(S.shape), V, time_dict


@timing
def calculate_predictions_version2(training_data, testing_data, dim_to_keep, N_test):
    """
    Det här s
    :param training_data:
    :param testing_data:
    :param dim_to_keep:
    :param N_test:
    :return:
    """

    # Center the data
    training_mean = training_data.mean(axis=0)
    testing_mean = testing_data.mean(axis=0)
    pixels_train_adj = training_data - training_mean
    pixels_test_adj = testing_data - training_mean

    U, S, Vt = np.linalg.svd(pixels_train_adj, full_matrices=False)
    V = Vt.T
    PCA_test_list = []
    y_pred_data = np.zeros([len(dim_to_keep), len(labels_test)])
    pred_rate = np.zeros(len(dim_to_keep))
    falses = np.zeros([len(N_dim_to_keep), N_test])

    time_dict = {}

    for i in range(len(dim_to_keep)):
        t0 = time.time()
        k = dim_to_keep[i]

        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        V_k = Vt[:k, :].T

        PCA = np.dot(U_k, S_k)
        PCA_test = pixels_test_adj.dot(V_k)
        PCA_test_list.append(PCA_test)

        n_correct = 0
        for j in range(len(labels_test)):
            print(f'Calculating: {i*len(labels_test) + j}/{len(dim_to_keep)*len(labels_test)}')
            diff_matrix = PCA - PCA_test[j, :]
            norm_matrix = np.linalg.norm(diff_matrix, axis=1)
            most_similar_index = np.where(norm_matrix == norm_matrix.min())[0][0]
            predicted_number = labels_train[most_similar_index]
            if predicted_number == labels_test[j]:
                n_correct += 1
            else:
                falses[i, predicted_number - 1] += 1

            y_pred_data[i, j] = predicted_number
        pred_rate[i] = n_correct / len(labels_test)
        time_dict[k] = (time.time() - t0)/N_test

    return PCA_test_list, y_pred_data, pred_rate, U, S, V, time_dict

@timing
def plot_predictions(PCA_test_list, V, dim_to_keep, N_test, n_train, show=True, save=False):
    for i in range(len(dim_to_keep)):
        k = dim_to_keep[i]
        pixel_data = np.matmul(PCA_test_list[i], V[:, :k].T)
        fig = plt.figure()
        Nx = 5
        Ny = math.ceil(N_test/Nx)
        for j in range(Nx * Ny):
            if pixel_data.shape[0] <= j:
                break
            plt.subplot(Ny, Nx, j + 1)
            plt.imshow(np.reshape(pixel_data[j, :], [28, 28]))
            plt.title(f'Correct: {labels_test[j]}\nPredicted: {int(y_pred_data[i, j])}')
            plt.axis('off')
        fig.suptitle(f'Number of principal components: {k}\nTraining set size {n_train}')
        if save:
            plt.savefig(f'saved_files/plot_k({k}).png')
        if show:
            plt.show()

def plot_single_prediction(PCA_test_list, S,  V, dim_to_keep, N_test, n_train, show=True, save=False):
    fig, axs = plt.subplots(1, len(dim_to_keep) + 1, figsize=(12, 4))
    import string
    for i, k in enumerate(dim_to_keep):
        pixel_data = np.matmul(PCA_test_list[i], V[:, :k].T)
        axs[i].imshow(np.reshape(pixel_data[1, :], [28, 28]))
        axs[i].axis('off')
        axs[i].set_title(f'{string.ascii_lowercase[i]}) k = {k}')
    axs[-1].plot(S, 'o', markersize=3, linewidth=0.5)
    axs[-1].set_box_aspect(True)
    # axs[-1].set_ylabel(r'$\sigma$')
    axs[-1].set_xlabel('k')
    axs[-1].set_title('d) Singularvärden')
    axs[-1].set_yticks([10000, 30000, 50000])
    axs[-1].set_yticklabels(['1', '3', '5'])
    axs[-1].set_xticks([20, 200, 400, 600])
    axs[-1].text(0.98, 0.95, r'$\sigma \times 10^3$', transform=axs[-1].transAxes, verticalalignment='top', horizontalalignment='right')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    if save:
        plt.savefig(f'saved_files/pcasiffror.png')
    if show:
        plt.show()
@timing
def plot_error_rate(pred_rate, dim_to_keep, n_test, n_train, show=True, save=False):
    fig, ax = plt.subplots(1, 1)
    ax.semilogx(dim_to_keep, pred_rate, 'x')
    ax.grid(True)
    ax.set_title(f'Prediction rate for {n_test} images trained on a set of {n_train} images')
    ax.set_xlabel('Nr. of principal components, k')
    ax.set_ylabel('Prediction rate')
    if save:
        plt.savefig(f'saved_files/prediction_rate({n_train}).png')
    if show:
        plt.show()


def plot_time_per_image(time_dict, dim_to_keep, n_train, show=True, save=False):
    fig, ax = plt.subplots(1, 1)
    x = np.array(dim_to_keep)
    y = np.array(list(time_dict.values()))*1000
    ax.scatter(x, y)
    k, m = np.polyfit(x, y, 1)
    ax.plot(x, k*x+m, label=f'{round(k,3)}x + {round(m,3)}', c='r')
    ax.set_title(f'Time per prediction with a training set of {n_train} images')
    ax.set_xlabel('Nr. of principal components, k')
    ax.set_ylabel('Time per prediction (ms)')
    ax.grid(True)
    ax.legend()
    if save:
        plt.savefig(f'saved_files/time_per_prediction_({n_train}).png')
    if show:
        plt.show()


def plot_prediction_per_time(pred_rate, time_dict, n_train, show=True, save=False):
    fig, ax = plt.subplots(1, 1)
    x = np.array(list(time_dict.values()))*1000
    y = np.array(pred_rate)
    ax.scatter(x, y)

    ax.set_title(f'Prediction rate vs time spent for a training set of {n_train} images')
    ax.set_xlabel('Time per prediction (ms)')
    ax.set_ylabel('Prediction rate')
    ax.grid(True)
    if save:
        plt.savefig(f'saved_files/prediction_vs_time({n_train}).png')
    if show:
        plt.show()


def plot_singular_values(S, n_train,  show=True, save=False):
    scs = S.cumsum()
    fig, ax = plt.subplots(1, 1)
    ax.plot(scs)
    # ax.semilogx(scs)
    ax.set_title('Cumsum of the k first singular values')
    ax.set_xlabel('Nr. of principal components, k')
    ax.set_ylabel(r'$ \sum^{k}_{n=0} \sigma _k $')
    if save:
        plt.savefig(f'saved_files/cumsum_singular_value({n_train}).png')
    if show:
        plt.show()


def result_table(dim_to_keep, time_dict, pred_rate, n_test, n_train, show=True, save=False):
    dict_ = {'index': dim_to_keep, 'time per frame (ms)': [float(v)*1000 for v in time_dict.values()], 'prediction_rate': pred_rate}
    df = pd.DataFrame(dict_)
    if show:
        print(df)
    if save:
        df.to_csv(f'saved_files/time_per_classification_training_set_size({n_train}).csv', sep=';')


if __name__ == '__main__':

    """
    test_images, labels = load_mnist(False, 20, normalize=False, keep_square=True)
    save_path = "C:\\Users\\albin\\Desktop\\MNIST rendered"
    for idx, image in enumerate(test_images):
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'image_{idx}.jpg'), bbox_inches='tight', pad_inches=0)
    print(labels)
    """

    # N_dim_to_keep = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 784]
    # N_dim_to_keep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 70, 90, 150, 300, 784]
    N_dim_to_keep = [5, 20, 100]
    # N_dim_to_keep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 70, 90, 150, 300, 784]

    N_train = 10000
    N_test = 1000
    pixels_train, labels_train = load_mnist(True, N_train, normalize=False)
    pixels_test, labels_test = load_mnist(False, N_test, normalize=False)

    n_test_imgs = len(pixels_test)
    n_train_imgs = len(pixels_train)
    # PCA_test_list, y_pred_data, pred_rate, U, S, V, time_dict = calculate_predictions(pixels_train, pixels_test, N_dim_to_keep, N_test)
    PCA_test_list, y_pred_data, pred_rate, U, S, V, time_dict = calculate_predictions_version2(pixels_train, pixels_test, N_dim_to_keep, N_test)

    numbers_to_plot = 10

    save_plots = True
    show_plots = True

    # plot_predictions(PCA_test_list, V, N_dim_to_keep, numbers_to_plot, n_train_imgs, show=show_plots, save=save_plots)
    plot_single_prediction(PCA_test_list, S, V, N_dim_to_keep, numbers_to_plot, n_train_imgs, show=show_plots, save=save_plots)

    # plot_error_rate(pred_rate, N_dim_to_keep, n_test_imgs, n_train_imgs, show=show_plots, save=save_plots)
    # plot_time_per_image(time_dict, N_dim_to_keep, n_train_imgs, show=show_plots, save=save_plots)
    # plot_singular_values(S, n_train_imgs, show=show_plots, save=save_plots)
    # plot_prediction_per_time(pred_rate, time_dict, n_train_imgs, show=show_plots, save=save_plots)
    # result_table(N_dim_to_keep, time_dict, pred_rate, n_test_imgs, n_train_imgs, show=show_plots, save=save_plots)
