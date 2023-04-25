import neural_network_training
import numpy as np
import pandas as pd
import extra_filer.extra_tools as tools
import os
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test_pca_matrix():
    """ A funtion thats trains and tests the models
        -> it is testing diffrent models depending on the parameters
        -> Saves the results in csv files
    """
    # Sets the parameters
    hidden_layer_sizes = [128, 256]
    k_list = [1, 2, 3, 5, 8, 14, 20, 28]
    # Defines varibles
    training_time_array    = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_time_array     = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_accuracy_array = np.zeros([len(hidden_layer_sizes), len(k_list)])
    # Loads the original data
    train_data, test_data = neural_network_training.get_MNIST()

    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        for j, k in enumerate(k_list):
            print(f'Training network with k={k}, hidden_layer_size={hidden_layer_size}')
            # Calls on test_pca_nn for train and test the nn
            testing_accuracy, training_time, testing_time = neural_network_training.test_pca_nn(k, epochs=10, hidden_layer_size=hidden_layer_size, train=train_data, test=test_data)
            # Stores the results to later save in a csv file
            training_time_array[i, j] = training_time
            testing_time_array[i, j] = testing_time
            testing_accuracy_array[i, j] = testing_accuracy
    # Writes to a csv file to save the data
    pd.DataFrame(training_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'training_time_pca.csv'), sep=';')
    pd.DataFrame(testing_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'testing_time_pca.csv'), sep=';')
    pd.DataFrame(testing_accuracy_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'testing_accuracy_pca.csv'), sep=';')

def test_score_matrix():

    hidden_layer_sizes = [128, 256]
    k_list = [1, 2, 3, 5, 8, 14, 20, 28]

    training_time_array    = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_time_array     = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_accuracy_array = np.zeros([len(hidden_layer_sizes), len(k_list)])

    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        for j, k in enumerate(k_list):
            print(f'Training network with k={k}, hidden_layer_size={hidden_layer_size}')
            training_time = neural_network_training.train_score_nn(k, hidden_layer_size=hidden_layer_size)
            testing_accuracy, testing_time = neural_network_training.test_score_nn(k, hidden_layer_size=hidden_layer_size)

            training_time_array[i, j] = training_time
            testing_time_array[i, j] = testing_time
            testing_accuracy_array[i, j] = testing_accuracy

    pd.DataFrame(training_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'training_time_pca.csv'), sep=';')
    pd.DataFrame(testing_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'testing_time_pca.csv'), sep=';')
    pd.DataFrame(testing_accuracy_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'testing_accuracy_pca.csv'), sep=';')

def plot_score_matrix():
    testing_accuracy = pd.read_csv(os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'testing_accuracy.csv'), sep=';')
    testing_time = pd.read_csv(os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'testing_time.csv'), sep=';')
    training_time = pd.read_csv(os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'training_time.csv'), sep=';')

    # We don't need to plot all of the rows/ different layer sizes
    selected_layer_sizes = [8, 16, 32, 128]
    testing_accuracy = testing_accuracy.loc[selected_layer_sizes]
    testing_time = testing_time.loc[selected_layer_sizes]
    training_time = training_time.loc[selected_layer_sizes]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    x = [int(v) for v in testing_accuracy.columns]
    labels = [f'lager1: {int(l)}, lager2: {int(l)//2}' for l in testing_accuracy.index]

    # Plotting the accuracy
    axs[0].plot(x, testing_accuracy.T, '-o')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Precision för olika lagerstorlekar och k')

    # Plotting the training time
    axs[1].plot(x, training_time.T, '-o')
    axs[1].set_ylabel('tid (s)')
    axs[1].set_title('Träningstid för olika lagerstorlekar och k')


    # Plotting the testing time
    testing_time = testing_time * 1000
    axs[2].plot(x, testing_time.T, '-o')
    axs[2].set_ylabel('tid (ms)')
    axs[2].set_title('Testtid för olika lagerstorlekar och k')

    for ax in axs:
        ax.set_xlabel('k')
        ax.legend(labels)
        ax.grid()

    fig.tight_layout()
    plt.show()
    print('hehe')

def plot_pca_matrix():
    """ taken from "plot_score_matrix" change som eparameters to fit the pca matrix
    -> plots the data
    """
    # Loads the already existing results from csv files
    testing_accuracy = pd.read_csv(os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'testing_accuracy_pca.csv'), sep=';', index_col=0)
    testing_time = pd.read_csv(os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'testing_time_pca.csv'), sep=';', index_col=0)
    training_time = pd.read_csv(os.path.join(neural_network_training.get_project_root(), 'neural_network_image_classification', 'results', 'training_time_pca.csv'), sep=';', index_col=0)
    # Creates the figure to plot in
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # Takes out the k values from the loaded results
    x = [int(v) for v in testing_accuracy.columns]
    # Takes out the layersize values from the loaded results
    labels = [f'{int(l)}, {int(l)//2}' for l in testing_accuracy.index]

    # Plotting the accuracy
    axs[0].plot(x, testing_accuracy.iloc[0], '-o')
    axs[0].plot(x, testing_accuracy.iloc[1], '-.v')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Precision för olika lagerstorlekar och k')

    # Plotting the training time
    axs[1].plot(x, training_time.iloc[0], '-o')
    axs[1].plot(x, training_time.iloc[1], '-.v')
    axs[1].set_ylabel('tid (s)')
    axs[1].set_title('Träningstid för olika lagerstorlekar och k')

    # Plotting the testing time
    testing_time = testing_time * 1000
    axs[2].plot(x, testing_time.iloc[0], '-o')
    axs[2].plot(x, testing_time.iloc[1], '-.v')
    axs[2].set_ylabel('tid (ms)')
    axs[2].set_title('Testtid för olika lagerstorlekar och k')

    for ax in axs:
        ax.set_xlabel('k')
        ax.legend(labels, title='Lagerstorlekar')
        ax.grid()

    for i, ax in enumerate(axs):
        ax.text(-0.14, 1.08, f"({chr(i + 97)})", transform=ax.transAxes,
                fontweight='bold', fontsize=12, va='top', ha='left')

    fig.tight_layout()
    plt.show()
    print('hehe')


if __name__ == '__main__':
    # test_pca_matrix()
    plot_pca_matrix()