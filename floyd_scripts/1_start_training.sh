#!/bin/sh

echo "\n - 1 - Start training"

export INPUT_PATH="/input"
export OUTPUT_PATH="/output"
git ad
cd $ROOT

python scripts/squeezenet_multilabel_classification_all_classes.py


echo "\n Finished training"