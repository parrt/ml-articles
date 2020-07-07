#!/bin/bash

BUILD=/tmp/regularization
WEB=/Users/parrt/github/website-explained.ai/regularization
SRC=/Users/parrt/github/ml-articles/regularization

# Support
mkdir -p $WEB
mkdir -p $WEB/css; cp -r $BUILD/css $WEB
mkdir -p $WEB/code; cp -r $SRC/code $WEB
cp ~/github/msds621/code/linreg/regularization_cloud.py $WEB/code

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
cp $SRC/images/lagrange-animation.gif $WEB/images
cp $SRC/images/L1L2contour.png $WEB/images
cp $SRC/images/L1contour.png $WEB/images
cp $SRC/images/L2contour.png $WEB/images

# Content
cp $BUILD/index.html $WEB
cp $BUILD/intro.html $WEB
cp $BUILD/constraints.html $WEB
cp $BUILD/impl.html $WEB
cp $BUILD/L1vsL2.html $WEB
