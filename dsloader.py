from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

# backslashes cause error in the string, so 'r' converts the string to raw string
# 800 png files
DATASET_PATH = r'C:\Users\kamil\Machine Learning\andy\image_data'
# 16 labels
CLASS_INDEX = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'slash', 'star']
NUM_CLASSES = 16
IMG_ROWS = 32
IMG_COLS = 32


def loadDataset():
    files = [f for f in listdir(DATASET_PATH)]
    features = np.empty((len(files), IMG_ROWS, IMG_COLS))
    labels = []

    for i, file in enumerate(files):
        feature = np.array(Image.open(DATASET_PATH + '\\' + file))
        features[i] = feature

        label = CLASS_INDEX.index(file.split('_')[0])  # labels 0 - 15
        labels.append(label)

    features = features / 255  # mean normalization of features
    return features, labels