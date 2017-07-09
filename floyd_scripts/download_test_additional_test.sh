#!/usr/bin/env sh

cd /output

# Install kaggle-cli
pip install -U kaggle-cli

kg download -u UUU -p XXX -c planet-understanding-the-amazon-from-space -f test-jpg-additional.tar.7z


##### install 7za
sudo apt-get -y update
sudo apt-get -y install p7zip-full unzip

mkdir -p /output/test

7z x /output/test-jpg-additional.tar.7z -o/output/test
tar -xf /output/test/test-jpg-additional.tar -C /output/test
mv /output/test/test-jpg-additional /output/test/jpg-additional
rm /output/test/test-jpg-additional.tar
rm /output/test-jpg-additional.tar.7z
