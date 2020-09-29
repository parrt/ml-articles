fswatch -o ~/github/ml-articles/tensor-sensor/index.html | xargs -n1 -I {} osascript -e 'tell application "Google Chrome" to tell the active tab of its first window to reload'
