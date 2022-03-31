import matplotlib.pyplot as plt
import pickle
def print_images(dataset, reconstructions,  anomaly_indices, show_anomalies):
  n = anomaly_indices if show_anomalies else [x for x in range(len(dataset)) if x not in anomaly_indices]
  
  plt.figure(figsize=(200, 4))

  counter = 0
  for i in n:
    # display original
    plt.title("reconstructed")
    ax = plt.subplot(2, len(n), counter + 1)
    plt.imshow(dataset[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    plt.title("reconstructed")
    ax = plt.subplot(2, len(n), len(n)+ 1 + counter)
    plt.imshow(reconstructions[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    counter +=1
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


# data = load_data(['out/reconstructions.txt','out/dataset.txt', 'out/anomaly_indices.txt'])
# print_images(data['out/dataset.txt'], data['out/reconstructions.txt'], data['out/anomaly_indices.txt'], show_anomalies=True)