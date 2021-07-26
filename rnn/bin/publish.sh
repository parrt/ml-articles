#!/bin/bash

BUILD=/tmp/rnn
WEB=/Users/parrt/github/website-explained.ai/rnn
SRC=/Users/parrt/github/ml-articles/rnn

# Support
mkdir -p $WEB
mkdir -p $WEB/css; cp -r $BUILD/css $WEB

# Images
mkdir -p $WEB/images
cp $BUILD/images/*.svg $WEB/images
cp $BUILD/images/*.png $WEB/images
cp $BUILD/images/*.gif $WEB/images

# Content
cp $BUILD/index.html $WEB
cp $BUILD/implementation.html $WEB
cp $BUILD/minibatch.html $WEB
