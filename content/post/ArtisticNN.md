---
title: "ArtisticNN"
date: 2017-07-15T10:27:55-05:00
---

# Introduction
This is an implementation of the artistic neural network as described in this [paper](www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). VGG pretrained model weights can be downloaded from [here](http://www.vlfeat.org/matconvnet/models/). Use imagenet-vgg-verydeep-19.mat and imagenet-vgg-verydeep-16.mat.


Checklist for dependencies:

* Tensorflow, preferably with GPU support
* numpy
* scipy
* PIL
* If you want to run in notebook, IPython notebook is also required.

# **Get the pretrained weights**
The following bash code will automatically download the models. VGG19 is approximately 510M and VGG16 is approximately 491M. Comment out using "#" if you already have the model downloaded somewhere else. But don't forget to change the model path later (you will see it).

```bash
if [ ! -e "models/imagenet-vgg-verydeep-19.mat" ]; then echo "VGG19 does not exist"; \
   wget -P models/ http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat; fi
if [ ! -e "models/imagenet-vgg-verydeep-16.mat" ]; then echo "VGG16 does not exist"; \
    wget -P models/ http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat; fi
```
