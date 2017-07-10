import os

os.environ['ROOT'] = '/output/planet_amazon_rainforest'
os.environ['INPUT_PATH'] = '/input'
os.environ['OUTPUT_PATH'] = '/output'

if __name__ == "__main__":

    os.system("echo \"----------- Run all scripts -----------\"")
    os.system("git clone --recursive https://github.com/vfdev-5/planet_amazon_rainforest $ROOT")
    os.system(" -- Finished Clone sources")
    os.system("echo \"---- Start validation predictions ----\"")
    os.system("cd $ROOT && python scripts/val_predict_squeezenet21_multilabel_classification_all_classes.py")
    os.system("echo \"Finished validation predictions\"")

