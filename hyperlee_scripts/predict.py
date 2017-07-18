import os
from datetime import datetime

os.environ['ROOT'] = '/home/user/DATA/planet_amazon_rainforest'
os.environ['INPUT_PATH'] = '/home/user/DATA/input'
os.environ['OUTPUT_PATH'] = '/home/user/DATA/output'
os.environ['KERAS_BACKEND'] = 'tensorflow'


if __name__ == "__main__":

    now = datetime.now()
    script_name='predict_resnet50_multilabel_classification_all_classes.py'
    log_filename='$OUTPUT_PATH/%s_%s.log' % (script_name, now.strftime("%Y%m%d_%H%M"))

    print("\n - 1 - Start predictions")
    os.system("cd $ROOT && python3 scripts/%s > %s 2>&1" % (script_name, log_filename))
    print("\n Finished predictions")
