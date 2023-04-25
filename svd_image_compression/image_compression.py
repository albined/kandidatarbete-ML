import os

from PIL import Image
import numpy as np
import pandas as pd
import extra_filer.extra_tools as tools

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def read_image(path='zelda.png'):
    img = Image.open(path)
    img_array = np.array(img)
    return img_array


def calculate_error_from_image_matrix_difference(original, compressed):
    error_matrix = original - compressed
    error = np.linalg.norm(error_matrix)
    return error


def calculate_error_from_remaining_singular_values(S, k):
    error = np.linalg.norm(S[k:])
    return error


if __name__ == '__main__':
    """
    Alear.jpg innehåller färre detaljer och tar därför mindre skada av att förlora
    singular values än vad donkey_kong.jpg gör 
    """
    # starting_image = 'alear.jpg'
    # starting_image = 'donkey_kong.jpg'
    starting_image = 'zelda.png'
    original_image = read_image(starting_image)

    k_list = [1, 5, 20, 100, 300, -1]

    error_dict = {}

    processed_image_list = []

    for k in k_list:

        # Perform SVD for all three color channels
        SVD_list = [np.linalg.svd(original_image[:, :, i], full_matrices=False) for i in range(3)]

        # Separate the U, S, and Vh components into their own lists
        U_list, S_list, Vh_list = list(map(list, zip(*SVD_list)))
        for idx, (u, s, v) in enumerate(zip(U_list, S_list, Vh_list)):
            U_list[idx] = u.astype(np.float32)
            S_list[idx] = s.astype(np.float32)
            Vh_list[idx] = v.astype(np.float32)

        # Multiply together only the k first singular values and vectors components to create a new image
        compressed_svd_list = [np.dot(U_list[i][:, :k] * S_list[i][:k], Vh_list[i][:k, :]) for i in range(3)]

        # Combine the color channels again and save it as a new image
        # Also clip the colors to 0 to 255 to prevent them from turning into a rainbow color.
        compressed_img_array = np.dstack(compressed_svd_list).clip(0, 255).astype('uint8')
        compressed_img = Image.fromarray(compressed_img_array)
        processed_image_list.append(compressed_img)
        save_path = os.path.join(tools.get_project_root(), 'svd_image_compression', 'compressed_images')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        compressed_img.save(os.path.join(save_path, f'ssscompressed_{starting_image.replace(".jpg", "").replace(".png", "")}_k({k}).png'))

        # Calculate the errors
        error = np.array([[calculate_error_from_image_matrix_difference(original_image[:, :, i], compressed_img_array[:, :, i]),
                           calculate_error_from_remaining_singular_values(S_list[i], k)] for i in range(3)])

        error_dict[k] = error.flatten()

    error_df = pd.DataFrame(error_dict).T
    error_df.columns = ['r, norm_delta_pixels', 'r, norm_remaining_singular_values',
                        'g, norm_delta_pixels', 'g, norm_remaining_singular_values',
                        'b, norm_delta_pixels', 'b, norm_remaining_singular_values']
    error_df.to_csv(f'Errors_{starting_image.replace(".jpg", "")}.csv', sep=';')
    print(error_df)


