
import os
from glob import glob
import pandas as pd
import numpy as np

project_common_path = os.path.dirname(__file__)

if 'INPUT_PATH' in os.environ:
    INPUT_PATH = os.environ['INPUT_PATH']
else:
    INPUT_PATH = os.path.abspath("../input")

DATA_PATH = INPUT_PATH

if 'OUTPUT_PATH' in os.environ:
    OUTPUT_PATH = os.environ['OUTPUT_PATH']
else:
    OUTPUT_PATH = os.path.abspath("../output")

TRAIN_DATA = os.path.join(DATA_PATH, "train")
TEST_DATA = os.path.join(DATA_PATH, "test")
TRAIN_CSV_FILEPATH = os.path.join(DATA_PATH, "train_v2.csv")
GENERATED_DATA = os.path.join(OUTPUT_PATH, 'generated')

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


equalized_data_classes = {0: ['selective_logging',
                              'slash_burn',
                              'blow_down',
                              'blooming',
                              'conventional_mine',
                              'artisinal_mine'],
                          1: ['bare_ground'],
                          2: ['haze',
                              'water',
                              'partly_cloudy',
                              'cultivation',
                              'road',
                              'habitation',
                              'cloudy'],
                          3: ['agriculture'],
                          4: ['primary', 'clear']
}


def create_data_class(k, row):
    tags = row.split(' ')
    ret = set(tags) & set(equalized_data_classes[k])
    if len(ret) > 0:
        return 1
    return 0


def create_image_id(image_name):
    return image_name[6:]


enc_cl_df_path = os.path.join(GENERATED_DATA, "train_enc_cl.csv")
if os.path.exists(enc_cl_df_path):
    TRAIN_ENC_CL_CSV = pd.read_csv(enc_cl_df_path)
else:
    TRAIN_ENC_CL_CSV = TRAIN_ENC_CSV.copy()
    for i in equalized_data_classes:
        TRAIN_ENC_CL_CSV['class_%i' % i] = TRAIN_ENC_CSV['tags'].apply(lambda x: create_data_class(i, x))

    TRAIN_ENC_CL_CSV["image_id"] = TRAIN_ENC_CSV["image_name"].apply(create_image_id)
    TRAIN_ENC_CL_CSV.to_csv(enc_cl_df_path, index=False)


def get_id_type_list_for_class(class_index):
    df = TRAIN_ENC_CL_CSV[TRAIN_ENC_CL_CSV['class_%i' % class_index] == 1]
    return get_id_type_list_from_df(df)


def get_id_type_list_from_df(df):
    return [(str(image_id), "Train_jpg") for image_id in df['image_id'].values]


def to_set(id_type_list):
    return set([(i[0], i[1]) for i in id_type_list])


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
    assert "Train" in image_type, "Can get only train caption"
    return TRAIN_CSV.loc[int(image_id), 'tags']


def get_label(image_id, image_type, as_series=False):
    assert "Train" in image_type, "Can get only train labels"
    if as_series:
        return TRAIN_ENC_CSV.loc[int(image_id), unique_tags]
    return TRAIN_ENC_CSV.loc[int(image_id), unique_tags].values


def get_class_label_mask(class_index):
    out = np.zeros(len(unique_tags), dtype=np.uint8)
    for name in equalized_data_classes[class_index]:
        out[unique_tags.index(name)] = 1
    return out


def find_best_weights_file(weights_files, field_name='val_loss', best_min=True):

    if best_min:
        best_value = 1e5
        comp = lambda a, b: a > b
    else:
        best_value = -1e5
        comp = lambda a, b: a < b

    if '=' != field_name[-1]:
        field_name += '='

    best_weights_filename = ""
    for f in weights_files:
        index = f.find(field_name)
        index += len(field_name)
        assert index >= 0, "Field name '%s' is not found in '%s'" % (field_name, f)
        end = f.find('_', index)
        val = float(f[index:end])
        if comp(best_value, val):
            best_value = val
            best_weights_filename = f
    return best_weights_filename, best_value


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


