#!/bin/bash

I="/Users/parrt/github/ml-articles/regularization/"
O="/tmp/regularization"

while true
do
	if test $I/css/article.css -nt $O/index.html || \
           test $I/index.xml -nt $O/index.html
	then
		java -jar /Users/parrt/github/bookish/target/bookish-1.0-SNAPSHOT.jar -target html -o $O $I/article.xml
	fi
	sleep .2s
done
