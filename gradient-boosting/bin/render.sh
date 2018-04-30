#!/bin/bash

I="/Users/parrt/github/ml-articles/gradient-boosting/"
O="/tmp/gradient-boosting"

while true
do
	if test $I/index.md -nt $O/index.html || test $I/css/article.css -nt $O/index.html
	then
		/Users/parrt/github/bookish/bin/article.sh $I/article.json
	fi
	sleep .2s
done
