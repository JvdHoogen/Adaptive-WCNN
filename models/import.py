## SCRIPT
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
# ### change the labels into categories
from tensorflow.keras.utils import to_categorical


import random as rn


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K



from tensorflow.keras.layers import Activation, Dense, Flatten, MaxPooling1D, Conv1D, AveragePooling1D
from tensorflow.keras.layers import Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import concatenate


def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

## Scale 3D array Input = 3D array, scalar object from sklearn. Output = scaled 3D array.
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

from tensorflow.keras.layers import Layer

