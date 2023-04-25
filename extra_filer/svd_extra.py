from matplotlib import pyplot as plt
import numpy as np
import pca_image_classification.image_recognition_pca as pca_tool


def calculate_all_singular_values_for_images(data, shape):
    singular_values = np.zeros((len(data), shape[0]))  # Om det är en osymmetrisk shape kanske det ska vara 1. IDK
    for i, row in enumerate(data):
        if i % 100 == 0: print(f'calculating svd {i}/{len(data)}')
        row = row.reshape(shape)
        U, S, V = np.linalg.svd(row)
        singular_values[i, :] = S
    return singular_values


def plot_all_singular_values(S):
    fig, ax = plt.subplots(1, 1)
    s_max = S.max(axis=0)
    s_min = S.min(axis=0)
    s_q25, s_q75 = np.quantile(S, (0.25, 0.75), axis=0)
    s_mean = S.mean(axis=0)

    # Plot max, min
    ax.plot(s_max, linestyle='--', color='green', label='max')
    ax.plot(s_min, linestyle='--', color='magenta', label='min')

    # Plot mean
    ax.plot(s_mean, lw=2, color='red', label='mean')

    # Plot shaded area between quartiles
    ax.fill_between(range(0, len(s_q25)), s_q25, s_q75, color='blue', alpha=0.2, label='IQR')

    # Add legend
    ax.legend()

    plt.title('Singular value distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('$\sigma$')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    # Hämta mnist data
    pixels_train, labels_train = pca_tool.load_mnist(True)

    # Beräkna SVD för alla bilder
    singular_values = calculate_all_singular_values_for_images(pixels_train, (28, 28))

    # Plotta värdena
    plot_all_singular_values(singular_values)

    # Gör också en plot för den kumulativa andelen som används täcks av de k första
    cumsum_sv = singular_values.cumsum(axis=1)
    normalized_cs_sv = cumsum_sv / cumsum_sv.max(axis=1)[:, None]
    plot_all_singular_values(normalized_cs_sv)
