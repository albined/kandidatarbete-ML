import os
from PIL import Image
import numpy as np
import extra_filer.extra_tools as tools


def read_image(path='zelda.png'):
    img = Image.open(path)
    img_array = np.array(img)
    return img_array


def compress_image(path, k):
    original_image = read_image(path)
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
    save_path = os.path.join(tools.get_project_root(), 'svd_image_compression', 'compressed_images')
    name = os.path.basename(path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, f'compressed_{name.replace(".jpg", "").replace(".png", "")}_k({k}).png')
    compressed_img.save(save_name)

    return save_name
