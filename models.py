import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, Model

class Autoencoder(Model):
  def __init__(self, encoder_layer_list, decoder_layer_list):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential(encoder_layer_list)
    self.decoder = tf.keras.Sequential(decoder_layer_list)
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class Encoder(Model):

  def __init__(self):
    super(Encoder, self).__init__()
    self.layers = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(128, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(25, activation="relu"),
    ])

  def call(self, inputs, training=None, mask=None):
    return self.layers(inputs)
