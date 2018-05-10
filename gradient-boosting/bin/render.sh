#!/bin/bash

I="/Users/parrt/github/ml-articles/gradient-boosting/"
O="/tmp/gradient-boosting"

while true
do
	if test $I/css/article.css -nt $O/L2-loss.html || \
           test $I/L2-loss.md -nt $O/L2-loss.html || \
           test $I/L1-loss.md -nt $O/L1-loss.html || \
           test $I/descent.md -nt $O/descent.html
	then
		/Users/parrt/github/bookish/bin/article.sh $I/article.json
	fi
	sleep .2s
done
