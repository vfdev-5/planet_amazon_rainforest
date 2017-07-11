import os

os.environ['ROOT'] = '/output/planet_amazon_rainforest'
os.environ['INPUT_PATH'] = '/input'
os.environ['OUTPUT_PATH'] = '/output'

if __name__ == "__main__":

    print("----------- Run all scripts -----------")
    print("\n - 0 - Clone sources")
    os.system("git clone --recursive https://github.com/vfdev-5/planet_amazon_rainforest $ROOT")
    print("\n -- Finished Clone sources")

    print("\n - 1 - Start training")
    os.system("cd $ROOT && python scripts/train_2step_squeezenet21_multilabel_classification_all_classes.py")
    print("\n Finished training")
