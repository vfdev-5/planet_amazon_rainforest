import os

import numpy as np

from data_utils import get_filename
from image_utils import imwrite, get_image_data


def create_glued_image(img1, img2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    r = np.random.rand()
    if r > 0.5:
        out = img1.copy()
        overlay = img2
    else:
        out = img2.copy()
        overlay = img1
    index = np.random.randint(0, 2)
    margin = 75
    c = out.shape[index ] //2

    # i = np.random.randint(c - margin, c + margin)     # Not really good random
    i = int(np.random.rand() * 2.0 * margin) + c - margin
    dim = out.shape[index] - i
    # j = np.random.randint(0, overlay.shape[index] - dim)     # Not really good random
    j = int(np.random.rand() * (overlay.shape[index] - dim))
    if index == 0:
        out[i:, :, :] = overlay[j: j +dim, :, :]
    else:
        out[:, i:, :] = overlay[:, j: j +dim, :]
    return out


def generate_glued_pairs(id_type_list, n_generated_files, seed=None):
    if seed is not None:
        np.random.seed(seed)

    counter = n_generated_files
    while counter > 0:
        ind1 = np.random.randint(len(id_type_list))
        ind2 = np.random.randint(len(id_type_list))
        if ind1 == ind2:
            ind2 = (ind2 + np.random.randint(1, len(id_type_list))) % len(id_type_list)

        assert id_type_list[ind1][1] == id_type_list[ind2][1], "Image type should be the same: %s != %s" % (
        id_type_list[ind1][1], id_type_list[ind2][1])
        img1 = get_image_data(*id_type_list[ind1])
        img2 = get_image_data(*id_type_list[ind2])
        img = create_glued_image(img1, img2)

        image_id = str(id_type_list[ind1][0]) + "_" + str(id_type_list[ind2][0])
        image_type = 'Generated_' + id_type_list[ind1][1]

        filepath = get_filename(image_id, image_type)
        if os.path.exists(filepath):
            # Do not overwrite existing
            continue
        imwrite(img, image_id, image_type)

        counter -= 1

