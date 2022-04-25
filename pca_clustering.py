import os
import tensorflow as tf
from tensorflow import keras
import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, losses
from keras.datasets import mnist
from keras.models import Model
from models import *
from utils import *
from packaging import version
import parameters_clustering as params
import math
import scipy.spatial.distance as dist
import random
from sklearn.decomposition import PCA

def reduce_features(img, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(img)

# os.system("rm -rf ./logs/")

# load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# pre-process data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# define test data
test_data = x_test
test_labels = y_test
test_classes, test_class_occurances = np.unique(test_labels, return_counts=True)
test_class_names = [str(x) for x in test_classes]

for norm_class in test_classes:

    print(f'PCA + K-Means Clustering for Normal Class: {norm_class}')

    # run PCA over all data
    train_reduced = reduce_features(x_train)
    test_reduced = reduce_features(x_test)

    # get normal and abnormal data splits
    norm_train_data, norm_train_labels = x_train[np.where(y_train == norm_class)], y_train[np.where(y_train == norm_class)]
    norm_test_data, norm_test_labels = x_test[np.where(y_test == norm_class)], y_test[np.where(y_test == norm_class)]

    # visualize data
    