#!/usr/bin/env sh

cd /output

# Install kaggle-cli
pip install -U kaggle-cli

kg download -u UUU -p XXX -c planet-understanding-the-amazon-from-space -f train_v2.csv.zip
kg download -u UUU -p XXX -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z
kg download -u UUU -p XXX -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z


##### install 7za
sudo apt-get -y update
sudo apt-get -y install p7zip-full unzip

##### Extract data from /input
unzip /output/train_v2.csv.zip -d /output
rm /output/train_v2.csv.zip

mkdir -p /output/train
mkdir -p /output/test

7z x /output/train-jpg.tar.7z -o/output/train
tar -xf /output/train/train-jpg.tar -C /output/train
mv /output/train/train-jpg /output/train/jpg
rm /output/train/train-jpg.tar
rm /output/train-jpg.tar.7z

7z x /output/test-jpg.tar.7z -o/output/test
tar -xf /output/test/test-jpg.tar -C /output/test
mv /output/test/test-jpg /output/test/jpg
rm /output/test/test-jpg.tar
rm /output/test-jpg.tar.7z


mkdir -p /output/test

7z x /output/test-jpg-additional.tar.7z -o/output/test
tar -xf /output/test/test-jpg-additional.tar -C /output/test
mv /output/test/test-jpg-additional /output/test/jpg-additional
rm /output/test/test-jpg-additional.tar
rm /output/test-jpg-additional.tar.7z
