import pandas as pd
import os
import extra_filer.extra_tools as tools
import matplotlib.pyplot as plt

def plot_results():
    line_styles = ['-', '--', '-.', ':', '',
              '--', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))]
    marker_styles = ['o', 'D', '^', '+', '>', '1', '2', '3',
           '4', 's', 'p', '*', 'h', 'H', 'v', 'x', 'D', 'd']
    testing_accuracy = pd.read_csv(os.path.join(tools.get_project_root(), 'pca_image_classification', 'results', f'accuracy.csv'), sep=';', index_col=0)
    testing_time = pd.read_csv(os.path.join(tools.get_project_root(), 'pca_image_classification', 'results', f'testing_time.csv'), sep=';', index_col=0)
    training_time = pd.read_csv(os.path.join(tools.get_project_root(), 'pca_image_classification', 'results', f'training_time.csv'), sep=';', index_col=0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    x = [int(v) for v in testing_accuracy.columns]

    special_labels = {
        -1: 'reducerad, 0.5',
        -2: 'reducerad, 0.75',
        -3: 'reducerad, 1'
    }
    labels = [f'{n}' if n not in special_labels else special_labels[n] for n in testing_accuracy.index]


    # Plotting the accuracy
    for idx, line in testing_accuracy.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[0].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx])
    axs[0].set_ylabel('Precision')
    # axs[0].set_ylim([0.5, 1])
    axs[0].set_title('Precision för olika stora träningsset och k')

    # Plotting the training time
    for idx, line in training_time.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[1].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx])
    axs[1].set_ylabel('tid (s)')
    # axs[1].set_ylim([0, 4])
    axs[1].set_title('Träningstid för olika stora träningsset och k')

    # Plotting the testing time
    testing_time = testing_time * 1000
    for idx, line in testing_time.reset_index().iterrows():
        idx = int(idx)
        line = line.values[1:]
        axs[2].plot(x, line, linestyle=line_styles[idx], marker=marker_styles[idx])
    axs[2].set_ylabel('tid (ms)')
    # axs[2].set_ylim([0, 3])
    axs[2].set_title('Testtid för olika stora träningsset och k')

    for ax in axs:
        ax.set_xlabel('k')
        ax.legend(labels, title='Träningsbilder')
        ax.grid()

    for i, ax in enumerate(axs):
        ax.text(-0.14, 1.08, f"({chr(i + 97)})", transform=ax.transAxes,
                fontweight='bold', fontsize=12, va='top', ha='left')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_results()