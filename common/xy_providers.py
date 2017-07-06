
import numpy as np
import cv2

from image_utils import get_image_data
from data_utils import get_caption, get_label, get_class_label_mask


def image_class_labels_provider(image_id_type_list,
                                image_size,
                                class_index,
                                channels_first=True,
                                test_mode=False,
                                seed=None,
                                cache=None,
                                verbose=0, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    class_label_mask = get_class_label_mask(class_index) if class_index is not None else None

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            key = (image_id, image_type)
            if cache is not None and key in cache:
                if verbose > 0:
                    print("-- Load from RAM")
                img, label = cache.get(key)

                if channels_first:
                    if img.shape[1:] != image_size[::-1]:
                        img = img.transpose([1, 2, 0])
                        img = cv2.resize(img, dsize=image_size[::-1])
                        img = img.transpose([2, 0, 1])
                else:
                    if img.shape[:2] != image_size[::-1]:
                        img = cv2.resize(img, dsize=image_size[::-1])
            else:
                if verbose > 0:
                    print("-- Load from disk")

                img = get_image_data(image_id, image_type)

                if img.shape[:2] != image_size:
                    img = cv2.resize(img, dsize=image_size)
                if channels_first:
                    img = img.transpose([2, 0, 1])

                img = img.astype(np.float32) / 255.0

                if class_index is not None:
                    label = get_label(image_id, image_type)
                    label = label * class_label_mask
                else:
                    label = None
                # fill the cache only at first time:
                if cache is not None and counter == 0:
                    cache.put(key, (img, label))

            if test_mode:
                yield img, label, (image_id, image_type)
            else:
                yield img, label

        if test_mode:
            return
        counter += 1


def image_label_provider(image_id_type_list,
                         image_size,
                         channels_first=True,
                         test_mode=False,
                         seed=None,
                         cache=None,
                         with_label=True,
                         verbose=0, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            key = (image_id, image_type)
            if cache is not None and key in cache:
                if verbose > 0:
                    print("-- Load from RAM")
                img, label = cache.get(key)

                if channels_first:
                    if img.shape[1:] != image_size[::-1]:
                        img = img.transpose([1, 2, 0])
                        img = cv2.resize(img, dsize=image_size[::-1])
                        img = img.transpose([2, 0, 1])
                else:
                    if img.shape[:2] != image_size[::-1]:
                        img = cv2.resize(img, dsize=image_size[::-1])
            else:

                if verbose > 0:
                    print("-- Load from disk")

                img = get_image_data(image_id, image_type)

                if img.shape[:2] != image_size:
                    img = cv2.resize(img, dsize=image_size)
                if channels_first:
                    img = img.transpose([2, 0, 1])

                img = img.astype(np.float32) / 255.0

                if with_label:
                    label = get_label(image_id, image_type)
                else:
                    label = None
                # fill the cache only at first time:
                if cache is not None and counter == 0:
                    cache.put(key, (img, label))

            if test_mode:
                yield img, label, (image_id, image_type)
            else:
                yield img, label

        if test_mode:
            return
        counter += 1


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
