class Utils:    
    import numpy as np
    import os
    import random
    from matplotlib import pyplot as plt
    import cv2
    import itertools
    
    from skimage.transform import resize
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    
    import tensorflow 
    import keras
    from keras import layers, models
    from keras.applications import EfficientNetB0
    from keras.utils import to_categorical
    from keras.layers import Dense
    from keras import optimizers
    
