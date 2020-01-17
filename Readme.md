# Emotion Classfier
This project implements Logistic Regression and Softmax Regression using numpy. Experiments are performed on CVPR2010_CK+ dataset.

## Usage
Simply run the script driver.py with optional parameters and there you go!
    ./driver.py -r reportPCA -d "./aligned/" -e "happiness, anger, disgust, fear, sadness, surprise"

Currently three optional arguments are required:
> -r ROUTINE, --routine ROUTINE # Run predefined routine, default to "test"
> -d DIR, --dir DIR # data directory, default to "./aligned/"
> -e EMOTIONS, --emotions EMOTIONS # delimited emotion classes, default to "happiness,anger"

Use option "-h" to get more help.