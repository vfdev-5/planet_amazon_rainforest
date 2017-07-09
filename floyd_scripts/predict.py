import os

os.environ['ROOT'] = '/output/planet_amazon_rainforest'
os.environ['INPUT_PATH'] = '/input'
os.environ['OUTPUT_PATH'] = '/output'

if __name__ == "__main__":

    os.system("echo \"----------- Run all scripts -----------\"")
    # Setup data environment
    os.system("mkdir -p /input/test")
    os.system("ln -s /input_1/test/jpg /input/test/jpg")
    os.system("ln -s /input_2/test/jpg-additional /input/test/jpg-additional")
    os.system("ls /input/test")

    os.system("echo \"- 0 - Clone sources\"")
    os.system("git clone --recursive https://github.com/vfdev-5/planet_amazon_rainforest $ROOT")
    os.system(" -- Finished Clone sources")

    os.system("echo \"- 1 - Start predictions\"")
    os.system("cd $ROOT && python scripts/predict_squeezenet21_multilabel_classification_all_classes.py")
    os.system("echo \"Finished predictions\"")
