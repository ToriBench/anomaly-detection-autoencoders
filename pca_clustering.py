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
from tensorboard.plugins import projector

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

    print(f'PCA for Normal Class: {norm_class}')

    # run PCA over all data
    train_embeddings = reduce_features([x.flatten() for x in x_train])
    test_embeddings = reduce_features([x.flatten() for x in x_test])

    # get normal and abnormal data splits
    norm_train_data, norm_train_labels = train_embeddings[np.where(y_train == norm_class)], y_train[np.where(y_train == norm_class)]
    norm_test_data, norm_test_labels = test_embeddings[np.where(y_test == norm_class)], y_test[np.where(y_test == norm_class)]

    # label anomalous (1) and normal (0) data
    norm_train_labels = [0 if x == norm_class else 1 for x in norm_train_labels]
    norm_test_labels = [0 if x == norm_class else 1 for x in norm_test_labels]

    # Set up a logs directory, so Tensorboard knows where to look for files.
    log_dir = os.path.join('./logs', f'projection_{norm_class}')
    os.makedirs(log_dir, exist_ok=True)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, f'metadata_{norm_class}.tsv'), "w") as f:
        for l in norm_test_labels:
            f.write(f"{int(l)}\n")

    # Create a checkpoint from embedding
    embeddings = tf.Variable(test_embeddings)
    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(os.path.join(log_dir, f"embedding_{norm_class}.ckpt"))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    # visualize data
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)
