#!/bin/bash
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ./data/ ..."
    cd ./data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

mkdir Lfw
cd Lfw

echo "Now downloading Figaro1k.zip ..."

wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz

echo "Unzip lfw-funneled.tgz ..."

tar -xvf lfw-funneled.tgz

echo "Now downloading GT Images ... "
wget wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz

echo "Unzip parts_lfw_funneled_gt_images.tgz ..."
tar -xvf parts_lfw_funneled_gt_images.tgz

echo "Now downloading txt files"
wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt
wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation.txt
wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt

echo "Making parts_train_val.txt ..."
cat parts_train.txt parts_validation.txt > parts_train_val.txt

echo "Finished!"
