#!/bin/bash

# WARNING: must edit index.html to remove unfinished chaps

BUILD=/tmp/regularization
WEB=/Users/parrt/github/website-explained.ai/regularization
SRC=/Users/parrt/github/ml-articles/regularization

# Support
mkdir -p $WEB
mkdir -p $WEB/css; cp -r $BUILD/css $WEB
mkdir -p $WEB/code; cp -r $BUILD/code $WEB

# Images
mkdir -p $WEB/images
cp $BUILD/images/*.svg $WEB/images
cp $SRC/images/l1-cloud.png $WEB/images
cp $SRC/images/l1-orthogonal-cloud.png $WEB/images
cp $SRC/images/l1-symmetric-cloud.png $WEB/images
cp $SRC/images/l2-cloud.png $WEB/images
cp $SRC/images/l2-orthogonal-cloud.png $WEB/images
cp $SRC/images/l2-symmetric-cloud.png $WEB/images
cp $SRC/images/ESL_reg.png $WEB/images

# Content
cp $BUILD/index.html $WEB
