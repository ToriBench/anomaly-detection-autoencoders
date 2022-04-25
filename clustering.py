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


def get_min_dist(X):
    dists = dist.cdist(X, X, metric='euclidean')
    np.fill_diagonal(dists, np.max(dists))
    return np.min(dists)


def get_random_point_in_n_sphere(n_dims, radius):
    # Return 'numberOfSamples' samples of vectors of dimension N
    # with an uniform distribution inside the N-Sphere of radius R.
    # RATIONALE: https://math.stackexchange.com/q/87238
    randomnessGenerator = np.random.default_rng()
    X = randomnessGenerator.normal(size=(1, n_dims))
    U = randomnessGenerator.random((1, 1))
    return (radius * U ** (1 / n_dims) / np.sqrt(np.sum(X ** 2, 1, keepdims=True)) * X)[0]


def get_random_point(center, radius, n_dims):

    # get random point in n-sphere of desired radius
    random_point_in_n_sphere = get_random_point_in_n_sphere(n_dims, radius)

    # translate point to center
    random_point = np.add(center, random_point_in_n_sphere)

    return random_point


def get_fake_encoded_points(x_encoded):
    x_encoded_fake = []
    min_dist = get_min_dist(x_encoded)
    fake_point_dist = max(0.0000001, min_dist - 0.0001)
    assert 0.0 < fake_point_dist < min_dist
    for x in x_encoded:
        x_fake = get_random_point(x, min_dist, n_dims=x_encoded.shape[1])
        x_encoded_fake.append(x_fake)
    return x_encoded_fake

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

# create model
encoder = Encoder()
encoder.compile(optimizer='adam', loss=params.train_loss_func, metrics=['accuracy'])
#
# x_train: 784
# x_train: 25
# x_train_fake = 25

# do first model pass to get initial encoded images
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)

# generate close fake points
x_train_encoded_fake = get_fake_encoded_points(x_train_encoded)
x_test_encoded_fake = get_fake_encoded_points(x_test_encoded)

# set target values
y_train = x_train_encoded_fake
y_test = x_test_encoded_fake

# define callbacks
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      min_delta=0,
      patience=0,
      verbose=0,
      mode='auto',
      baseline=None,
      restore_best_weights=False
  )
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train
history = encoder.fit(x_train, y_train,
                      epochs=params.epochs,
                      validation_data=(y_train, y_test),
                      callbacks=[tensorboard_callback])

# save model
path = 'encoder_models'
os.makedirs(path, exist_ok=True)
encoder.save(path)

# test
test_loss_ave = 0.0
for i in range(len(x_test)):
    y_true = y_test[i]
    y_pred = encoder.predict(x_test[i])
    test_loss = tf.keras.losses.mean_squared_error(y_true[i], y_pred[i])
    test_loss_ave += test_loss
test_loss_ave /= len(x_test)

