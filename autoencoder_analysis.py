
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
from parameters import *

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

test_data = x_test
test_labels = y_test

test_classes, test_class_occurances = np.unique(test_labels, return_counts=True)
test_class_names = [str(x) for x in test_classes]

number_of_classes = len(test_classes)
cm = np.zeros((number_of_classes, number_of_classes))  # Confusion matrix
accuracy_matrix = []

def plot_accuracy(test_labels, anomaly_predictions, norm_class):
  results = []
  for i in range(10):
    lookup = np.where(test_labels==i)
    pred_label = anomaly_predictions[lookup]
    if i == norm_class:
      true_label = np.zeros(lookup[0].shape[0])
    else:
      true_label = np.ones(lookup[0].shape[0])
    accuracy = sum(pred_label == true_label)/len(lookup[0])
    results.append(round(accuracy,2))
  return results

for norm_class in test_classes:
    train_filter_norm = np.where(y_train == norm_class)
    test_filter_norm = np.where(y_test == norm_class)

    norm_train_data, norm_train_labels = x_train[train_filter_norm], y_train[train_filter_norm]
    norm_test_data, norm_test_labels = x_test[test_filter_norm], y_test[test_filter_norm]


    path = f'models/{norm_class}/{epochs}_epochs_{train_loss_func}_trainloss_{test_loss_func}_testloss_{dirty_data_percentage}_dirtydata'
    autoencoder = keras.models.load_model(path)

    # Determine threshold from inputting normal test data
    reconstructions = autoencoder.predict(norm_test_data)
    losses = tf.keras.losses.mean_squared_error(tf.reshape(norm_test_data, [norm_test_data.shape[0], 784]), tf.reshape(reconstructions, [reconstructions.shape[0], 784]))

    # losses = np.zeros(len(norm_train_data))
    # for i in range(len(norm_train_data)):
    #     losses[i] = loss_function(norm_train_data[i], reconstructions[i]).numpy()
    multiplier = 0.5 if (norm_class == 2) else ( 0.25 if (norm_class == 8) else 1)
    threshold = np.mean(losses) + multiplier * np.std(losses)

    reconstructions = autoencoder.predict(test_data)

    anomaly_indices = []
    number_of_inputs = len(test_data)
    number_of_classes = len(test_classes)
    anomaly_predictions = np.zeros(number_of_inputs)

    test_loss = 0

    for i in range(number_of_inputs):
        test_loss = tf.keras.losses.mean_squared_error(tf.reshape(test_data[i],[784]), tf.reshape(reconstructions[i],[784]))

        if np.any(tf.math.greater(test_loss,threshold).numpy()):
            anomaly_predictions[i] = 1
            anomaly_indices.append(i)
            
    # plt.hist(test_loss[None, :], bins=50)

    # plt.xlabel("Test loss")

    # plt.ylabel("No of examples")

    # plt.show()
    anomaly_label = np.where(y_test == norm_class, 0 ,1)
    accuracy_matrix.append(plot_accuracy(test_labels, anomaly_predictions, norm_class))

    accuracy = sum(anomaly_predictions == anomaly_label)/len(test_data)
    print(f"Class: {norm_class} Accuracy: {accuracy}")

    # filtered = filter(lambda x: x<= 49, anomaly_indices)
    # print_images(test_data[0:50], reconstructions[0:50],list(filtered), show_anomalies=False)
    


cm_figure = plot_confusion_matrix(np.array(accuracy_matrix), class_names=test_class_names,
                      title='Accuracy of each model per class', 
                      axis_names=['Inputs', 'Trained Models']) 