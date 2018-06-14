#!/bin/bash

I="/Users/parrt/github/ml-articles/gradient-boosting/"
O="/tmp/gradient-boosting"

while true
do
	if test $I/css/article.css -nt $O/L2-loss.html || \
           test $I/L2-loss.xml -nt $O/L2-loss.html || \
           test $I/L1-loss.xml -nt $O/L1-loss.html || \
           test $I/faq.xml -nt $O/faq.html || \
           test $I/descent.xml -nt $O/descent.html
	then
		java -jar /Users/parrt/github/bookish/target/bookish-1.0-SNAPSHOT.jar -target html -o /tmp/gradient-boosting $I/article.xml
	fi
	sleep .2s
done
