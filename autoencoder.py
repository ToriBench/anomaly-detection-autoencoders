# %load_ext tensorboard
import tensorflow as tf
from tensorflow import keras
import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, losses
from keras.datasets import mnist
from keras.models import Model
import os
from models import *
import pickle
from utils import *

os.system("rm -rf ./logs/") 
(x_train, y_train), (x_test, y_test)  = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

train_filter_norm = np.where(y_train == 1)
test_filter_norm = np.where(y_test == 1)
test_filter_anom = np.where(y_test == 1)

norm_train_data, norm_train_labels = x_train[train_filter_norm], y_train[train_filter_norm]
norm_test_data, norm_test_labels = x_test[test_filter_norm], y_test[test_filter_norm]

anom_test_data = x_test[test_filter_anom]

encoder_layer_list = [
      layers.Flatten(),
      layers.Dense(128, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu")
    ]

decoder_layer_list = [
      layers.Dense(64, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(784, activation="sigmoid"),
      layers.Reshape((28, 28))
    ]

autoencoder_dnn = Autoencoder(encoder_layer_list,decoder_layer_list)

autoencoder_dnn.compile(optimizer='adam', loss='binary_crossentropy')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = autoencoder_dnn.fit(norm_train_data, norm_train_data, 
          epochs=25, 
          validation_data=(norm_test_data, norm_test_data),
          callbacks=[tensorboard_callback])

encoder_layer_list = [
      layers.Conv2D(32,(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same')
    ]

decoder_layer_list = [
      layers.UpSampling2D((2, 2)),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
      layers.UpSampling2D((2, 2)),
      layers.Conv2D(1, (3, 3), activation='relu',padding='same'),
      layers.Reshape((28,28))
]

# autoencoder_cnn = Autoencoder(encoder_layer_list=encoder_layer_list, decoder_layer_list=decoder_layer_list)

# autoencoder_cnn.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder_cnn.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

earlyStop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False
)

# history = autoencoder_cnn.fit(norm_train_data, norm_train_data, 
#           epochs=16, 
#           validation_data=(norm_test_data, norm_test_data),
#           callbacks=[tensorboard_callback, earlyStop_callback])

autoencoder = autoencoder_dnn
reconstructions = autoencoder.predict(norm_test_data)
bce = tf.keras.losses.BinaryCrossentropy()
losses = np.zeros(len(norm_test_data))
print(bce(norm_test_data, reconstructions).numpy())
for i in range(len(norm_test_data)):
  losses[i]=bce(norm_test_data[i], reconstructions[i]).numpy()

print(np.mean(losses))

threshold = np.mean(losses) + 2 * np.std(losses)
print(threshold)

dataset = anom_test_data
reconstructions = autoencoder.predict(dataset)
bce = tf.keras.losses.BinaryCrossentropy()
anomaly_indices = []
test_loss = np.zeros(dataset.shape[0])
for i in range(dataset.shape[0]-1):
  test_loss=bce(dataset[i], reconstructions[i]).numpy()
  if test_loss > threshold:
    anomaly_indices.append(i)
counter = 0

print('Number of anomalies detected: ', len(anomaly_indices))
print('Number of anomalous inputs: ', len(dataset))
print('Accuracy: ',(len(anomaly_indices)/len(dataset)))

os.makedirs('out', exist_ok=True)
save_data(
  {
    'out/dataset.txt': dataset,
    'out/reconstructions.txt': reconstructions,
    'out/anomaly_indices. txt': anomaly_indices
  }
)
print_images(dataset, reconstructions, anomaly_indices, show_anomalies=True)

