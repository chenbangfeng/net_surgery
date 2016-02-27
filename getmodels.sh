#!/usr/bin/env bash

if [ ! -d "caffe_models" ]; then
  mkdir caffe_models
fi

echo "Downloading caffe models ..."

cd caffe_models

if [ -f "bvlc_alexnet.caffemodel" ]; then
  rm bvlc_alexnet.caffemodel
fi
wget https://www.dropbox.com/s/stj16abgc1xusxs/bvlc_alexnet.caffemodel

if [ -f "bvlc_alexnet_conv.caffemodel" ]; then
  rm bvlc_alexnet_conv.caffemodel
fi
wget https://www.dropbox.com/s/2szc09llne1e8zx/bvlc_alexnet_conv.caffemodel

if [ -f "bvlc_alexnet_full_conv_992.caffemodel" ]; then
  rm bvlc_alexnet.caffemodel
fi
wget https://www.dropbox.com/s/0i6q7ghpswi24a6/bvlc_alexnet_full_conv_992.caffemodel

if [ -f "bvlc_alexnet_full_conv_1000.caffemodel" ]; then
  rm bvlc_alexnet.caffemodel
fi
wget https://www.dropbox.com/s/bog262vrouvgqlq/bvlc_alexnet_full_conv_1000.caffemodel


# ----------------------------------------------------------------------
echo "Downloading torch models ..."

cd ..
if [ ! -d "torch_models" ]; then
  mkdir torch_models
fi
cd torch_models


if [ -f "alexnet_full_conv.net" ]; then
  rm alexnet_full_conv.net
fi
wget https://www.dropbox.com/s/wmwx8j3zrihh1z5/alexnet_full_conv.net

if [ -f "alexnet.net" ]; then
  rm alexnet.net
fi
wget https://www.dropbox.com/s/me62x47iyzog7f5/alexnet.net

if [ -f "alexnet_full_conv_992.net" ]; then
  rm alexnet_full_conv_992.net
fi
wget https://www.dropbox.com/s/xmbmkkosajpiapw/alexnet_full_conv_992.net

if [ -f "alexnet_full_conv_1000.net" ]; then
  rm alexnet_full_conv_1000.net
fi
wget https://www.dropbox.com/s/tsotmvgfk4fe47w/alexnet_full_conv_1000.net

if [ -f "th_model_fc6_bin8.net" ]; then
    rm th_model_fc6_bin8.net
fi
    wget https://www.dropbox.com/s/1bk784soe23hxly/th_model_fc6_bin8.net


if [ -f "th_model_fc7_bin8.net" ]; then
    rm th_model_fc7_bin8.net
fi
    wget https://www.dropbox.com/s/tb9uyi1rdtck0n8/th_model_fc7_bin8.net


#echo "Unzipping..."
#tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz

echo "Done."

