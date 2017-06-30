
import numpy as np
import cv2

from image_utils import get_image_data, get_caption


def image_caption_provider(image_id_type_list,
                           image_size,
                           channels_first=True,
                           test_mode=False,
                           seed=None,
                           with_caption=True,
                           verbose=0):

    if seed is not None:
        np.random.seed(seed)

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            if verbose > 0:
                print("-- Load from disk")

            img = get_image_data(image_id, image_type)

            if img.shape[:2] != image_size:
                img = cv2.resize(img, dsize=image_size)
            if channels_first:
                img = img.transpose([2, 0, 1])

            img = img.astype(np.float32) / 255.0

            if with_caption:
                cap = get_caption(image_id, image_type)
            else:
                cap = None

            if test_mode:
                yield img, cap, (image_id, image_type)
            else:
                yield img, cap

        if test_mode:
            return
        counter += 1
