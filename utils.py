import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import itertools
import numpy as np


def print_images(dataset, reconstructions,  anomaly_indices, show_anomalies):
    n = anomaly_indices if show_anomalies else [
        x for x in range(len(dataset)) if x not in anomaly_indices]
    print(n)

    plt.figure(figsize=(400, 4))

    counter = 0
    for i in n:
        # display original
        plt.title("orig.")
        ax = plt.subplot(2, len(n), counter + 1)
        plt.imshow(dataset[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        plt.title("recons.")
        ax = plt.subplot(2, len(n), len(n) + 1 + counter)
        plt.imshow(reconstructions[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        counter += 1
    plt.show()


def save_data(dictionary):
    for file, data in dictionary.items():
        with open(file, 'wb') as fp:
            pickle.dump(data, fp)


def load_data(filenames):
    dictionary = dict()

    for filename in filenames:
        with open(filename, 'rb') as fp:
            dictionary[filename] = pickle.load(fp)

    return dictionary


# def create_output_matrices(matrix, trained_model_class_index, classes, dataset, dataset_labels, number_of_label_occurances, reconstructions, loss_function, threshold):
#     """
#     Returns a 2D matrix containing the fraction of anomalies picked up.

#     Args:
#       cm (array, shape = [n, n]): a confusion matrix of integer classes
#       class_names (array, shape = [n]): String names of the integer classes
#     """
#     anomaly_indices = []
#     number_of_inputs = len(dataset)
#     number_of_classes = len(classes)
#     anomaly_predictions = np.zeros(number_of_inputs)
#     cm = np.zeros((number_of_classes, number_of_classes))  # Confusion matrix
#     test_loss = 0

#     for i in enumerate(dataset):
#         test_loss = loss_function(dataset[i], reconstructions[i]).numpy()
#         if test_loss > threshold:
#             anomaly_predictions[i] = 1
#             anomaly_indices.append(i)

#     for count, value in enumerate(classes):
#         input_indices = np.where(dataset_labels == value)
#         prediction = sum(anomaly_predictions[input_indices])
#         cm[trained_model_class_index][count] = round(
#             prediction/number_of_label_occurances[count], 2)

    return cm, anomaly_indices


def plot_confusion_matrix(cm, class_names, title, axis_names, path):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = cm

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel(axis_names[1])
    plt.xlabel(axis_names[0])

    plt.savefig(path)
    plt.show()

    return figure
