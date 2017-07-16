#!/usr/bin/env sh

INPUT_PATH=~/DATA/input
USER=
PSWD=

cd $INPUT_PATH

# Install kaggle-cli
#pip install -U kaggle-cli

#kg download -u $USER -p $PSWD -c planet-understanding-the-amazon-from-space -f train_v2.csv.zip
#kg download -u $USER -p $PSWD -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z
#kg download -u $USER -p $PSWD -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z
#kg download -u $USER -p $PSWD -c planet-understanding-the-amazon-from-space -f test-jpg-additional.tar.7z

##### install 7za
#sudo apt-get -y update
#sudo apt-get -y install p7zip-full unzip

##### Extract data from /input
unzip $INPUT_PATH/train_v2.csv.zip -d $INPUT_PATH
rm $INPUT_PATH/train_v2.csv.zip

mkdir -p $INPUT_PATH/train
mkdir -p $INPUT_PATH/test

7z x $INPUT_PATH/train-jpg.tar.7z -o$INPUT_PATH/train
tar -xf $INPUT_PATH/train/train-jpg.tar -C $INPUT_PATH/train
mv $INPUT_PATH/train/train-jpg $INPUT_PATH/train/jpg
rm $INPUT_PATH/train/train-jpg.tar
rm $INPUT_PATH/train-jpg.tar.7z

7z x $INPUT_PATH/test-jpg.tar.7z -o$INPUT_PATH/test
tar -xf $INPUT_PATH/test/test-jpg.tar -C $INPUT_PATH/test
mv $INPUT_PATH/test/test-jpg $INPUT_PATH/test/jpg
rm $INPUT_PATH/test/test-jpg.tar
rm $INPUT_PATH/test-jpg.tar.7z


mkdir -p $INPUT_PATH/test

7z x $INPUT_PATH/test-jpg-additional.tar.7z -o$INPUT_PATH/test
tar -xf $INPUT_PATH/test/test-jpg-additional.tar -C $INPUT_PATH/test
mv $INPUT_PATH/test/test-jpg-additional $INPUT_PATH/test/jpg-additional
rm $INPUT_PATH/test/test-jpg-additional.tar
rm $INPUT_PATH/test-jpg-additional.tar.7z

mkdir -p $INPUT_PATH/test

7z x $INPUT_PATH/test-jpg-additional.tar.7z -o$INPUT_PATH/test
tar -xf $INPUT_PATH/test/test-jpg-additional.tar -C $INPUT_PATH/test
mv $INPUT_PATH/test/test-jpg-additional $INPUT_PATH/test/jpg-additional
mv $INPUT_PATH/test/jpg-additional/*.jpg $INPUT_PATH/test/jpg/
rm -R $INPUT_PATH/test/jpg-additional
rm $INPUT_PATH/test/test-jpg-additional.tar
rm $INPUT_PATH/test-jpg-additional.tar.7z
