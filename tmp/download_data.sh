#!/bin/sh


pip install -U kaggle-cli

kg download -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z
kg download -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z

