import os

import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector
from sklearn.model_selection import train_test_split

model_id = '20220408-142503'

model = tf.keras.models.load_model('saved_model/' + model_id)

model.encoder.summary()

dataframe = pd.read_csv(
    'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

_, test_data, _, test_labels = train_test_split(
    data, labels, test_size=0.2
)

min_val = tf.reduce_min(test_data)
max_val = tf.reduce_max(test_data)

test_data = (test_data - min_val) / (max_val - min_val)
test_data = tf.cast(test_data, tf.float32)

print(test_data.shape)

# train_labels = train_labels.astype(bool)
# test_labels = test_labels.astype(bool)


# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir = os.path.join('./logs', model_id, 'projection')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for l in test_labels:
        f.write(f"{int(l)}\n")

embeddings = tf.Variable(model.encoder(test_data), name='features')

# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
# checkpoint = tf.train.Checkpoint(embedding=model.encoder)
checkpoint = tf.train.Checkpoint(embedding=embeddings)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))


# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)
