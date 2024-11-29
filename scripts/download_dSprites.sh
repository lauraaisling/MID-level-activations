#!/bin/bash
# coding=utf-8
echo "Downloading dSprites dataset."
if [[ ! -d "dsprites" ]]; then
  mkdir data
  mkdir data/dsprites
  wget -O data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
  echo "Downloading dSprites completed!"
fi