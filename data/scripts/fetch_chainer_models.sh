#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=chainer_models.tgz
URL="https://drive.google.com/uc?id=0B9P1L--7Wd2vVnpjcEVONGRiMUE&export=download"

wget $URL -O $FILE

echo "Unzipping file..."

tar zxvf $FILE

echo "Done."
