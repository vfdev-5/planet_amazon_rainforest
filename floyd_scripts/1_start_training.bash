#!/usr/bin/env bash

echo "\n - 1 - Start training"



pushd $ROOT

python scripts/squeezenet_multilabel_classification_all_classes.py

popd

echo "\n Finished training"