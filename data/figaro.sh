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

echo "Now downloading Figaro1k.zip ..."

# wget http://projects.i-ctm.eu/sites/default/files/AltroMateriale/207_Michele%20Svanera/Figaro1k.zip
# The official link is not working for some reason, so temporarily use dropbox instead.

wget https://www.dropbox.com/s/35momrh68zuhkei/Figaro1k.zip

echo "Unzip Figaro1k.zip ..."

unzip Figaro1k.zip

echo "Removing unnecessary files ..."

rm -f Figaro1k.zip
rm -f Figaro1k/GT/Training/*'(1).pbm'
rm -f Figaro1k/.DS_Store
rm -rf __MACOSX

echo "Finished!"
