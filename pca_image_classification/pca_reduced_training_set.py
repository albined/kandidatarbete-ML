import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
import math


def import_mnist():
    test_data = pd.read_csv("mnist_test_10.csv", delimiter=',', header=None).to_numpy()
    xdata_test = test_data[:, 1:]
    ydata_test = test_data[:, 0]

    train_data = pd.read_csv("mnist_train.csv", delimiter=',', header=None).to_numpy()
    train_data[np.isnan(train_data)] = 0
    train_data = train_data[train_data[:, 0].argsort()]

    pos_index = np.zeros(11, dtype=int) - 1
    train_data_mean = np.zeros((10, len(train_data[0, :])), dtype=int)

    plt.figure(num=1)
    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.1)
    plt.suptitle("New training set", fontsize=20, y=0.95, fontname="Helvetica")

    for i in range(0, 10):
        pos_index[i + 1] = np.where(train_data[:, 0] == i)[0][-1]
        dummy = train_data[pos_index[i] + 1: pos_index[i+1], :]
        train_data_mean[i, :] = dummy.mean(axis=0)

        nr_of_images_x = 5
        nr_of_images_y = math.ceil(len(train_data_mean) / nr_of_images_x)
        ax = plt.subplot(nr_of_images_y, nr_of_images_x, i + 1)
        ax.imshow(np.reshape(train_data_mean[i, 1:], (28, 28)), cmap="cividis")
        ax.axis('off')
        ax.set_title(f"Label: %i" % train_data_mean[i, 0], fontsize=10, loc='center', fontname="Helvetica")

    xdata_train = train_data_mean[:, 1:]
    ydata_train = train_data_mean[:, 0]

    return xdata_test, ydata_test, xdata_train, ydata_train,


def pca_function(xdata_test, xdata_train):
    xdata_mean = xdata_train.mean(axis=0)

    xdata_test_adj = xdata_test - xdata_mean
    xdata_train_adj = xdata_train - xdata_mean

    c = np.cov(xdata_train_adj, rowvar=False)
    u, s, v_t = linalg.svd(c)
    v = np.transpose(v_t)

    return xdata_test_adj, xdata_train_adj, v


def image_recognition(k, xdata_test_adj, xdata_train_adj, v, ydata_test, ydata_train):

    predicted_label = np.zeros((len(k), len(ydata_test)), dtype=int)

    n_t = np.zeros(len(k))

    for i in range(len(k)):
        pca_train = np.matmul(xdata_train_adj, v[:, :k[i]])
        pca_test = np.matmul(xdata_test_adj, v[:, :k[i]])

        plt.figure(num=i+2)
        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(wspace=0.15)
        plt.suptitle(f"Testdata då k = %i" % k[i], fontsize=18, y=0.95, fontname="Helvetica")

        xdata_test_plot = np.matmul(pca_test, np.transpose(v[:, :k[i]]))

        for j in range(len(ydata_test)):
            min_d = 1e6
            for z in range(len(ydata_train)):
                d = np.linalg.norm(pca_test[j, :] - pca_train[z, :])
                if d < min_d:
                    min_d = d
                    predicted_label[i, j] = ydata_train[z]

            if predicted_label[i, j] == ydata_test[j]:
                n_t[i] = n_t[i] + 1

            nr_of_images_x = 5
            nr_of_images_y = math.ceil(len(ydata_test)/nr_of_images_x)
            ax = plt.subplot(nr_of_images_y, nr_of_images_x, j + 1)
            ax.imshow(np.reshape(xdata_test_plot[j, :], (28, 28)), cmap="cividis")
            ax.axis('off')
            ax.set_title("Predicted label: \nCorrect label: ", fontsize=8, loc='left', fontname="Helvetica")
            ax.set_title("{f1}\n{f2}".format(f1=predicted_label[i, j], f2=ydata_test[j]), fontsize=8, loc='right',
                         fontname="Helvetica")

    plt.figure(num=len(k)+2)
    plt.semilogx(k, np.divide(n_t, len(ydata_test)), ".", color="black", markerfacecolor='none',
                 ms=10, markeredgecolor='black')
    plt.xlabel("Nr. of principal components, k")
    plt.ylabel("Prediction Rate")
    plt.grid(True, which="both")
    plt.title("Prediction rate for 10 images")
    plt.rc('font', size=12)
    plt.show()


if __name__ == '__main__':
    Xdata_test, Ydata_test, Xdata_train, Ydata_train = import_mnist()
    Xdata_test_adj, Xdata_train_adj, V = pca_function(Xdata_test, Xdata_train)

    k_factor = np.array([1, 2, 3, 4, 5, 6, 12, 25, 50, 100, 200, 784])
    image_recognition(k_factor, Xdata_test_adj, Xdata_train_adj, V, Ydata_test, Ydata_train)
