fswatch -o /tmp/regularization/L1vsL2.html | xargs -n1 -I {} osascript -e 'tell application "Google Chrome" to tell the active tab of its first window to reload'
