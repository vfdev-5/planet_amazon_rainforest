#!/usr/bin/env bash

echo "\n - 1 - Start training"

export INPUT_DATA="/input"
export OUTPUT_DATA="/output"

pushd $ROOT

python scripts/squeezenet_multilabel_classification_all_classes.py

popd

echo "\n Finished training"