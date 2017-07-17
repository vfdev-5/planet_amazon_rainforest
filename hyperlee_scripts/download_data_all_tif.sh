#!/usr/bin/env sh

INPUT_PATH=~/DATA/input
USER=
PSWD=

cd $INPUT_PATH

# Install kaggle-cli
#pip install -U kaggle-cli

#kg download -u $USER -p $PSWD -c planet-understanding-the-amazon-from-space -f train-tif-v2.tar.7z
#kg download -u $USER -p $PSWD -c planet-understanding-the-amazon-from-space -f test-tif-v2.tar.7z

##### install 7za
#sudo apt-get -y update
#sudo apt-get -y install p7zip-full unzip

##### Extract data from /input
7z x $INPUT_PATH/train-tif-v2.tar.7z -o$INPUT_PATH/train
tar -xf $INPUT_PATH/train/train-tif-v2.tar -C $INPUT_PATH/train
mv $INPUT_PATH/train/train-tif-v2 $INPUT_PATH/train/tif
rm $INPUT_PATH/train/train-tif-v2.tar
rm $INPUT_PATH/train-tif-v2.tar.7

7z x $INPUT_PATH/test-tif-v2.tar.7z -o$INPUT_PATH/test
tar -xf $INPUT_PATH/test/test-tif-v2.tar -C $INPUT_PATH/test
mv $INPUT_PATH/test/test-tif-v2 $INPUT_PATH/test/tif
rm $INPUT_PATH/test/test-tif-v2.tar
rm $INPUT_PATH/test-tif-v2.tar.7z
