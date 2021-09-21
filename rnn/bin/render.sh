#!/bin/bash

I="/Users/parrt/github/ml-articles/rnn/"
O="/tmp/rnn"

mkdir -p $O

while true
do
	if test $I/index.html -nt $O/index.html || \
	   test $I/implementation.xml -nt $O/implementation.html || \
           test $I/minibatch.xml -nt $O/minibatch.html
	then
		cp $I/index.html $O
		java -jar /Users/parrt/.m2/repository/us/parr/bookish/1.0-SNAPSHOT/bookish-1.0-SNAPSHOT.jar -target html -o $O $I/article.xml
	fi
	sleep .2s
done
