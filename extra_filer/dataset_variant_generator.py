from extra_filer.extra_tools import append_matrix_to_csv, load_mnist, EasyTimeLog
from sklearn.decomposition import PCA
import numpy as np
import os

timelog = EasyTimeLog()


def create_pca_library(train_set, label_train, test_set, label_test, principal_components, save_folder=None,
                       dataset_name='', normalize=True):
    # Define the pca, it's much easier if we just use a library
    tl = timelog.start_log(f'PCA on the training set with {len(train_set)} images', f'k={principal_components}')
    pca = PCA(n_components=principal_components)
    pca.fit(train_set)

    pca_train = pca.transform(train_set)
    mean = 0
    std = 1
    if normalize:
        mean = pca_train.mean()
        std = pca_train.std()
        pca_train = (pca_train - mean) / std
    tl.stop()

    tl = timelog.start_log(f'PCA projection of {len(test_set)} testing images', f'k={principal_components}')
    pca_test = pca.transform(test_set)
    if normalize:
        print(f'normalizing mean: {mean}, std: {std}')
        pca_test = (pca_test - mean) / std
    tl.stop()

    if save_folder is not None:
        append_matrix_to_csv(pca_train,
                             f'pca_transformed_training_{dataset_name}_k={principal_components}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder,
                                                 f'pca_transformed_{dataset_name}_k={principal_components}'))
        append_matrix_to_csv(pca_test,
                             f'pca_transformed_testing_{dataset_name}_k={principal_components}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder,
                                                 f'pca_transformed_{dataset_name}_k={principal_components}'))
        append_matrix_to_csv(label_train,
                             f'labels_svd_transformed_training_{dataset_name}_k={principal_components}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder,
                                                 f'pca_transformed_{dataset_name}_k={principal_components}'))
        append_matrix_to_csv(label_test,
                             f'labels_svd_transformed_testing_{dataset_name}_k={principal_components}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder,
                                                 f'pca_transformed_{dataset_name}_k={principal_components}'))


def create_svd_libaries(train_set, label_train, test_set, label_test, truncation: int, save_folder=None,
                        dataset_name='', normalize=True):
    U_train = np.zeros((train_set.shape[0], train_set.shape[1] * truncation))
    Vt_train = np.zeros((train_set.shape[0], train_set.shape[2] * truncation))
    k = truncation
    tl = timelog.start_log(f'svd on the training set with {len(train_set)} images', f'k={k}')
    Umean = 0
    Vtmean = 0
    Ustd = 1
    Vtstd = 1
    for i, img in enumerate(train_set):
        u, _, vt = np.linalg.svd(img)
        v = vt.T
        u_k = u[:, :k]
        v_k = v[:, :k]
        U_train[i, :] = u_k.flatten()
        Vt_train[i, :] = v_k.flatten()
    if normalize:
        Umean = U_train.mean()
        Vtmean = Vt_train.mean()
        Ustd = U_train.std()
        Vtstd = Vt_train.std()
        U_train = (U_train - Umean) / Ustd
        Vt_train = (Vt_train - Vtmean) / Vtstd
    tl.stop()

    U_test = np.zeros((train_set.shape[0], train_set.shape[1] * truncation))
    Vt_test = np.zeros((train_set.shape[0], train_set.shape[2] * truncation))
    tl = timelog.start_log(f'svd on the testing set with {len(test_set)} images', f'k={k}')
    for i, img in enumerate(test_set):
        u, _, vt = np.linalg.svd(img)
        v = vt.T
        u_k = u[:, :k]
        v_k = v[:, :k]
        U_test[i, :] = u_k.flatten()
        Vt_test[i, :] = v_k.flatten()
    if normalize:
        U_test = (U_test - Umean) / Ustd
        Vt_test = (Vt_test - Vtmean) / Vtstd
    tl.stop()

    if save_folder is not None:
        # Save for U
        append_matrix_to_csv(U_train, f'U_SVD_transformed_training_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'U_svd_transformed_{dataset_name}_k={k}'))
        append_matrix_to_csv(U_test, f'U_SVD_transformed_testing_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'U_svd_transformed_{dataset_name}_k={k}'))
        append_matrix_to_csv(label_train, f'labels_SVD_transformed_training_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'U_svd_transformed_{dataset_name}_k={k}'))
        append_matrix_to_csv(label_test, f'labels_SVD_transformed_testing_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'U_svd_transformed_{dataset_name}_k={k}'))
        # Save for V
        append_matrix_to_csv(Vt_train, f'Vt_SVD_transformed_training_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'V_svd_transformed_{dataset_name}_k={k}'))
        append_matrix_to_csv(Vt_test, f'Vt_SVD_transformed_testing_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'V_svd_transformed_{dataset_name}_k={k}'))
        append_matrix_to_csv(label_train, f'labels_SVD_transformed_training_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'V_svd_transformed_{dataset_name}_k={k}'))
        append_matrix_to_csv(label_test, f'labels_SVD_transformed_testing_{dataset_name}_k={k}.csv',
                             relative_root=True,
                             folder=os.path.join(save_folder, f'V_svd_transformed_{dataset_name}_k={k}'))


def pca_library_main(k, dataset_name='MNIST'):
    n_test = -1  # all
    pixels_train, labels_train = load_mnist(True, n_test)
    pixels_test, labels_test = load_mnist(False, n_test)
    create_pca_library(pixels_train, labels_train, pixels_test, labels_test, k,
                       save_folder='Generated dataset variants', dataset_name=dataset_name)


def svd_libraries_main(k, dataset_name='MNIST'):
    n_test = -1  # all
    pixels_train, labels_train = load_mnist(True, n_test, True)
    pixels_test, labels_test = load_mnist(False, n_test, True)
    create_svd_libaries(pixels_train, labels_train, pixels_test, labels_test, k,
                        save_folder='Generated dataset variants', dataset_name=dataset_name)
