# IFT 6756
Personal repo for Game Theory and ML course.

Course webpage: https://gauthiergidel.github.io/courses/game_theory_ML_2021.html

***

# Final Project

## Data
 * QuickDraw data - download from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

## Installing DiffVG
See https://github.com/BachiLi/diffvg
```
git clone git@github.com:BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y pytorch torchvision -c pytorch
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```