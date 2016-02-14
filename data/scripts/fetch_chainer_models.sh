#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=chainer_models.tgz
URL="https://drive.google.com/uc?id=0B9P1L--7Wd2vWjM1T28tMC1OSTQ&export=download"

gdown $URL -O $FILE

echo "Unzipping file..."

tar zxvf $FILE

echo "Done."
