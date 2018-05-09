#!/bin/bash

I="/Users/parrt/github/ml-articles/gradient-boosting/"
O="/tmp/gradient-boosting"

while true
do
	if test $I/css/article.css -nt $O/L2-norm.html || \
           test $I/L2-norm.md -nt $O/L2-norm.html || \
           test $I/L1-norm.md -nt $O/L1-norm.html || \
           test $I/descent.md -nt $O/descent.html
	then
		/Users/parrt/github/bookish/bin/article.sh $I/article.json
	fi
	sleep .2s
done
