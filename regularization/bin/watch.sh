#fswatch -o /tmp/regularization/impl.html | xargs -n1 -I {} osascript -e 'tell application "Google Chrome" to tell the active tab of its first window to reload'
fswatch -o ~/github/ml-articles/regularization/index.html | xargs -n1 -I {} osascript -e 'tell application "Google Chrome" to tell the active tab of its first window to reload'
