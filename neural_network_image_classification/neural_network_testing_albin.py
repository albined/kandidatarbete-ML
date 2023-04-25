import neural_network_training
import numpy as np
import pandas as pd
import extra_filer.extra_tools as tools
import os
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def test_score_matrix(onlytest=False):

    hidden_layer_sizes = [8, 16, 32, 64, 128, 256]
    k_list = [1, 2, 3, 5, 8, 14, 20, 28, 64, 128]

    training_time_array    = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_time_array     = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_accuracy_array = np.zeros([len(hidden_layer_sizes), len(k_list)])

    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        for j, k in enumerate(k_list):
            print(f'Testing network with k={k}, hidden_layer_size={hidden_layer_size}')
            if not onlytest:
                training_time = neural_network_training.train_score_nn(k, hidden_layer_size=hidden_layer_size)
            testing_accuracy, testing_time = neural_network_training.test_score_nn(k, hidden_layer_size=hidden_layer_size, tests=10)

            if not onlytest:
                training_time_array[i, j] = training_time
            testing_time_array[i, j] = testing_time
            testing_accuracy_array[i, j] = testing_accuracy

    if not onlytest:
        pd.DataFrame(training_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
            os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'score_training_time.csv'), sep=';')
    pd.DataFrame(testing_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'score_testing_time.csv'), sep=';')
    pd.DataFrame(testing_accuracy_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'score_testing_accuracy.csv'), sep=';')

def test_u_v_matrix(u_v, onlytest=False):

    hidden_layer_sizes = [16, 64, 256]
    k_list = [1, 2, 3, 5, 8, 14, 20] # Max k=28 på den här eftersom bilden är max 28x28 px

    training_time_array    = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_time_array     = np.zeros([len(hidden_layer_sizes), len(k_list)])
    testing_accuracy_array = np.zeros([len(hidden_layer_sizes), len(k_list)])

    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        for j, k in enumerate(k_list):
            print(f'Testing network with k={k}, hidden_layer_size={hidden_layer_size}')
            if not onlytest:
                training_time = neural_network_training.train_svd_nn(k, u_v, hidden_layer_size=hidden_layer_size)
            testing_accuracy, testing_time = neural_network_training.test_svd_nn(k, u_v, hidden_layer_size=hidden_layer_size, tests=10)

            if not onlytest:
                training_time_array[i, j] = training_time
            testing_time_array[i, j] = testing_time
            testing_accuracy_array[i, j] = testing_accuracy

    if not onlytest:
        pd.DataFrame(training_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
            os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{u_v.lower()}_training_time.csv'), sep=';')
    pd.DataFrame(testing_time_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{u_v.lower()}_testing_time.csv'), sep=';')
    pd.DataFrame(testing_accuracy_array, columns=k_list, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{u_v.lower()}_testing_accuracy.csv'), sep=';')


def test_normal_matrix(onlytest=False):

    hidden_layer_sizes = [8, 16, 32, 64, 128, 256]

    training_time_array    = np.zeros([len(hidden_layer_sizes)])
    testing_time_array     = np.zeros([len(hidden_layer_sizes)])
    testing_accuracy_array = np.zeros([len(hidden_layer_sizes)])

    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        print(f'Testing network with hidden_layer_size={hidden_layer_size}')
        if not onlytest:
            training_time = neural_network_training.train_normal_mnist_nn(hidden_layer_size=hidden_layer_size)
        testing_accuracy, testing_time = neural_network_training.test_normal_mnist_nn(hidden_layer_size=hidden_layer_size, tests=10)

        if not onlytest:
            training_time_array[i] = training_time
        testing_time_array[i] = testing_time
        testing_accuracy_array[i] = testing_accuracy

    if not onlytest:
        pd.DataFrame(training_time_array, index=np.array(hidden_layer_sizes)).to_csv(
            os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'normal_training_time.csv'), sep=';')
    pd.DataFrame(testing_time_array, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'normal_testing_time.csv'), sep=';')
    pd.DataFrame(testing_accuracy_array, index=np.array(hidden_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'normal_testing_accuracy.csv'), sep=';')


def test_cnn_matrix(onlytest=False):

    linear_layer_sizes = [8, 16, 32, 64, 128, 256]
    convolutional_layer_sizes = [8, 16, 32]

    training_time_array    = np.zeros([len(linear_layer_sizes), len(convolutional_layer_sizes)])
    testing_time_array     = np.zeros([len(linear_layer_sizes), len(convolutional_layer_sizes)])
    testing_accuracy_array = np.zeros([len(linear_layer_sizes), len(convolutional_layer_sizes)])

    for i, linear_layer_size in enumerate(linear_layer_sizes):
        for j, convolutional_layer_size in enumerate(convolutional_layer_sizes):
            print(f'Testing network with fcc={linear_layer_size}, cn2d={convolutional_layer_size}')
            if not onlytest:
                training_time = neural_network_training.train_cnn_mnist_nn(linear_layer_size=linear_layer_size,
                                                                           convolutional_layer_size=convolutional_layer_size)
            testing_accuracy, testing_time = neural_network_training.test_cnn_mnist_nn(linear_layer_size=linear_layer_size, convolutional_layer_size=convolutional_layer_size, tests=10)

            if not onlytest:
                training_time_array[i, j] = training_time
            testing_time_array[i, j] = testing_time
            testing_accuracy_array[i, j] = testing_accuracy

    if not onlytest:
        pd.DataFrame(training_time_array, columns=convolutional_layer_sizes, index=np.array(linear_layer_sizes)).to_csv(
            os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'cnn_training_time.csv'), sep=';')
    pd.DataFrame(testing_time_array, columns=convolutional_layer_sizes, index=np.array(linear_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'cnn_testing_time.csv'), sep=';')
    pd.DataFrame(testing_accuracy_array, columns=convolutional_layer_sizes, index=np.array(linear_layer_sizes)).to_csv(
        os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', 'cnn_testing_accuracy.csv'), sep=';')


def plot_result_matrix(test_name='score'):
    line_styles = ['-', '--', '-.', ':',
              '--', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))]
    marker_styles = ['o', 'D', '^', '+', '>', '1', '2', '3',
           '4', 's', 'p', '*', 'h', 'H', 'v', 'x', 'D', 'd']
    testing_accuracy = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{test_name}_testing_accuracy.csv'), sep=';', index_col=0)
    testing_time = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{test_name}_testing_time.csv'), sep=';', index_col=0)
    training_time = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{test_name}_training_time.csv'), sep=';', index_col=0)

    # We don't need to plot all of the rows/ different layer sizes
    selected_layer_sizes = [16, 32, 128, 256]
    selected_layer_sizes = [16, 64, 256]
    testing_accuracy = testing_accuracy.loc[selected_layer_sizes]
    testing_time = testing_time.loc[selected_layer_sizes]
    training_time = training_time.loc[selected_layer_sizes]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    x = [int(v) for v in testing_accuracy.columns]

    labels = [f'{int(l)}, {int(l)//2}' for l in testing_accuracy.index]

    # Plotting the accuracy
    for idx, line in testing_accuracy.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[0].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx])
    axs[0].set_ylabel('Precision')
    axs[0].set_ylim([0.5, 1])
    axs[0].set_title('Precision för olika lagerstorlekar och k')

    # Plotting the training time
    for idx, line in training_time.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[1].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx])
    axs[1].set_ylabel('tid (s)')
    axs[1].set_ylim([8, 12.5])
    axs[1].set_title('Träningstid för olika lagerstorlekar och k')

    # Plotting the testing time
    testing_time = testing_time * 10**6
    for idx, line in testing_time.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[2].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx])
    axs[2].set_ylabel('tid (µs)')
    axs[2].set_ylim([7, 12])
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


def plot_normal_result():
    testing_accuracy = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{"normal"}_testing_accuracy.csv'), sep=';', index_col=0)
    testing_time = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{"normal"}_testing_time.csv'), sep=';', index_col=0)
    training_time = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{"normal"}_training_time.csv'), sep=';', index_col=0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    x = testing_time.index
    xl = x.delete(1)
    xticklabels = [f'{l}\n{l//2}' for l in xl]

    # Plotting the accuracy
    axs[0].plot(x, testing_accuracy.T.iloc[0], '-o')
    axs[1].set_ylabel('precision')
    axs[0].set_ylim([0.85, 1])
    axs[0].set_title('Precision för olika lagerstorlekar')

    # Plotting the training time
    axs[1].plot(x, training_time.T.iloc[0], '-o')
    axs[1].set_ylabel('tid (s)')
    axs[1].set_ylim([0, 20])
    axs[1].set_title('Träningstid för olika lagerstorlekar')

    # Plotting the testing time
    testing_time = testing_time * 10**6
    axs[2].plot(x, testing_time.T.iloc[0], '-o')
    axs[2].set_ylabel('tid (µs)')
    axs[2].set_ylim([0, 20])
    axs[2].set_title('Testtid för olika lagerstorlekar')

    for ax in axs:
        ax.set_xlabel('Lagerstorlekar')
        ax.set_xticks(xl, xticklabels)
        ax.grid()

    for i, ax in enumerate(axs):
        ax.text(-0.14, 1.08, f"({chr(i + 97)})", transform=ax.transAxes,
                fontweight='bold', fontsize=12, va='top', ha='left')

    fig.tight_layout()
    plt.show()


def plot_cnn_result():
    line_styles = ['-', '--', '-.', ':',
              '--', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))]
    marker_styles = ['o', 'D', '^', '+', '>', '1', '2', '3',
           '4', 's', 'p', '*', 'h', 'H', 'v', 'x', 'D', 'd']
    testing_accuracy = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{"cnn"}_testing_accuracy.csv'), sep=';', index_col=0).T
    testing_time = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{"cnn"}_testing_time.csv'), sep=';', index_col=0).T
    training_time = pd.read_csv(os.path.join(tools.get_project_root(), 'neural_network_image_classification', 'results', f'{"cnn"}_training_time.csv'), sep=';', index_col=0).T

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))


    x = [int(v) for v in testing_accuracy.columns]
    labels = [f'{n}, {int(n)*2}' for n in testing_accuracy.index]

    # Plotting the accuracy
    for idx, line in testing_accuracy.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[0].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx], label=labels[idx])
    # axs[0].plot(x, testing_accuracy.T.iloc[0], '-o')
    axs[1].set_ylabel('precision')
    axs[0].set_ylim([0.94, 1])
    axs[0].set_title('Precision för olika lagerstorlekar')

    # Plotting the training time
    for idx, line in training_time.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[1].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx], label=labels[idx])
    # axs[1].plot(x, training_time.T.iloc[0], '-o')
    axs[1].set_ylabel('tid (s)')
    axs[1].set_ylim([11, 15])
    axs[1].set_title('Träningstid för olika lagerstorlekar')

    # Plotting the testing time
    testing_time = testing_time * 10**6
    for idx, line in testing_time.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[2].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx], label=labels[idx])
    # axs[2].plot(x, testing_time.T.iloc[0], '-o')
    axs[2].set_ylabel('tid (µs)')
    axs[2].set_ylim([9, 13])
    axs[2].set_title('Testtid för olika lagerstorlekar')

    for ax in axs:
        ax.set_xlabel('Storlek på linjärt lager')
        # ax.set_xticks(xl, xticklabels)
        ax.grid()
        ax.legend(title='Storlek faltningslager')

    for i, ax in enumerate(axs):
        ax.text(-0.14, 1.08, f"({chr(i + 97)})", transform=ax.transAxes,
                fontweight='bold', fontsize=12, va='top', ha='left')

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    # test_score_matrix(onlytest=True)
    # test_u_v_matrix('V')
    # test_u_v_matrix('U')
    # test_normal_matrix()
    # test_cnn_matrix(onlytest=False)
    plot_result_matrix('score')
    # plot_result_matrix('U')
    # plot_result_matrix('V')
    # plot_normal_result()
    # plot_cnn_result()
