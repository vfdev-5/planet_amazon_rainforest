#!/bin/sh

echo "----------- Run all scripts -----------"

export ROOT=/output/planet_amazon_rainforest

echo "\n - 0 - Clone sources"

git clone --recursive https://github.com/vfdev-5/planet_amazon_rainforest $ROOT

echo "\n -- Finished Clone sources"

echo "\n - 1 - Start training"

export INPUT_PATH="/input"
export OUTPUT_PATH="/output"

cd $ROOT

python scripts/squeezenet2_multilabel_classification_all_classes.py


echo "\n Finished training"
