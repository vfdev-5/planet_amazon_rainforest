
import os
from glob import glob
import pandas as pd
import numpy as np

project_common_path = os.path.dirname(__file__)

DATA_PATH = os.path.abspath("../input")
INPUT_PATH = DATA_PATH


TRAIN_DATA = os.path.join(DATA_PATH, "train")
TEST_DATA = os.path.join(DATA_PATH, "test")
TRAIN_CSV_FILEPATH = os.path.join(DATA_PATH, "train_v2.csv")
GENERATED_DATA = os.path.join(INPUT_PATH, 'generated')

if not os.path.exists(GENERATED_DATA):
    os.makedirs(GENERATED_DATA)

TRAIN_CSV = pd.read_csv(TRAIN_CSV_FILEPATH)

train_jpg_files = glob(os.path.join(TRAIN_DATA, "jpg", "*.jpg"))
train_jpg_ids = [s[len(os.path.join(TRAIN_DATA, "jpg"))+1+len('train_'):-4] for s in train_jpg_files]

test_jpg_files = glob(os.path.join(TRAIN_DATA, "jpg", "*.jpg"))
test_jpg_ids = [s[len(os.path.join(TRAIN_DATA, "jpg"))+1+len('test_'):-4] for s in test_jpg_files]


def get_unique_tags(df):
    unique_tags = set()
    image_tags = df['tags'].apply(lambda x: x.split(' '))
    for line in image_tags:
        for l in line:
            unique_tags.add(l)
    return list(unique_tags)

unique_tags = get_unique_tags(TRAIN_CSV)

tag_to_index = {}
for i, l in enumerate(unique_tags):
    tag_to_index[l] = i


def encode_tags(df):
    unique_tags = get_unique_tags(df)
    n_tags = len(unique_tags)

    def tags2vec(tags):
        enc = [tag_to_index[tag] for tag in tags.split(" ")]
        out = np.array([0] * n_tags)
        out[enc] = 1
        return out.tolist()

    enc_df = TRAIN_CSV.copy()
    for c in unique_tags:
        enc_df.loc[:, c] = np.zeros((len(TRAIN_CSV), 1), dtype=np.uint8)
    enc_df.loc[:, unique_tags] = np.array(TRAIN_CSV['tags'].apply(tags2vec).tolist())
    return enc_df

enc_df_path = os.path.join(GENERATED_DATA, "train_enc.csv")
if os.path.exists(enc_df_path):
    TRAIN_ENC_CSV = pd.read_csv(enc_df_path)
else:
    TRAIN_ENC_CSV = encode_tags(TRAIN_CSV)
    TRAIN_ENC_CSV.to_csv(enc_df_path, index=False)
    

def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type
    """

    check_dir = False
    if "Train" in image_type:
        ext = 'jpg' if 'jpg' in image_type else 'tif'
        data_path = os.path.join(TRAIN_DATA, ext)
        prefix = 'train_'
    elif "Test" in image_type:
        ext = 'jpg' if 'jpg' in image_type else 'tif'
        data_path = os.path.join(TEST_DATA, ext)
        prefix = 'test_'
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}{}.{}".format(prefix, image_id, ext))


def get_caption(image_id, image_type):
    assert "Train" in image_type, "Can get only train labels"
    return TRAIN_CSV.loc[int(image_id), 'tags']


class DataCache(object):
    """
    Queue storage of any data to avoid reloading
    """
    def __init__(self, n_samples):
        """
        :param n_samples: max number of data items to store in RAM
        """
        self.n_samples = n_samples
        self.cache = {}
        self.ids_queue = []

    def put(self, data_id, data):

        if 0 < self.n_samples < len(self.cache):
            key_to_remove = self.ids_queue.pop(0)
            self.cache.pop(key_to_remove)

        self.cache[data_id] = data
        if data_id in self.ids_queue:
            self.ids_queue.remove(data_id)
        self.ids_queue.append(data_id)

    def get(self, data_id):
        return self.cache[data_id]

    def remove(self, data_id):
        self.ids_queue.remove(data_id)
        self.cache.pop(data_id)

    def __contains__(self, key):
        return key in self.cache and key in self.ids_queue


