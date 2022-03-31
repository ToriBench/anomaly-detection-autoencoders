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


