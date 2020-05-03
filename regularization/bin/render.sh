#!/bin/bash

I="/Users/parrt/github/ml-articles/regularization/"
O="/tmp/regularization"

while true
do
	cp $I/index.html $O
	if test $I/intro.xml -nt $O/intro.html || \
           test $I/constraints.xml -nt $O/constraints.html || \
           test $I/impl.xml -nt $O/impl.html || \
           test $I/L1vsL2.xml -nt $O/L1vsL2.html
	then
		java -jar /Users/parrt/github/bookish/target/bookish-1.0-SNAPSHOT.jar -target html -o $O $I/article.xml
	fi
	sleep .2s
done
