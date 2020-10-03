#!/bin/bash

BUILD=/tmp/tensor-sensor
WEB=/Users/parrt/github/website-explained.ai/tensor-sensor
SRC=/Users/parrt/github/ml-articles/tensor-sensor

# Support
mkdir -p $WEB
mkdir -p $WEB/css; cp -r $BUILD/css $WEB

# Images
mkdir -p $WEB/images
cp $BUILD/images/*.svg $WEB/images
cp $BUILD/images/*.png $WEB/images

# Content
cp $BUILD/index.html $WEB
