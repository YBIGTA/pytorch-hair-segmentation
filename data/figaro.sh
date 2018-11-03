#!/bin/bash

echo "Now downloading Figaro1k.zip ..."

wget http://projects.i-ctm.eu/sites/default/files/AltroMateriale/207_Michele%20Svanera/Figaro1k.zip

echo "Unzip Figaro1k.zip ..."

unzip Figaro1k.zip

echo "Removing unnecessary files ..."

rm -f Figaro1k.zip
rm -f Figaro1k/GT/Training/*'(1).pbm'
rm -rf __MACOSX

echo "Finished!"
