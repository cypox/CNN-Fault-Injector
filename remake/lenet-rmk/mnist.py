import os
from random import shuffle

class dataset:
  def __init__(self, dataset, label_file):
    self.dataset = dataset
    self.labels = label_file

    with open(self.labels, 'r') as f:
      content = f.read()
      self.items = content.split('\n')
      self.items.remove('')

    print('[\033[95mdataset \033[0m]: loading labels')
    self.labels = dict()
    with open(label_file, 'r') as f:
      for line in f:
        line.replace('\n', '')
        if line != '':
          key = line.split(' ')[0]
          value = int(line.split(' ')[1])
          self.labels[key] = value
  
  def is_correct_prediction(self, ikey, output):
    top_1 = 0
    if self.labels[ikey] == output[0].argmax():
      top_1 = 1
    return top_1

  def get_subset(self, max_size = None, max_occurences = 1):
    filenames = os.listdir(self.dataset)
    shuffle(self.items)

    picked = []
    subset = []

    if max_size is not None:
      at_most_size = max_size
    else:
      at_most_size = 1000 * max_occurences

    for line in self.items:
      if len(subset) >= at_most_size:
        break
      line = line.replace('\n', '')
      # print('{}'.format(line))
      file_ = line.split(' ')[0]
      class_ = line.split(' ')[1]
      if picked.count(class_) < max_occurences:
        subset.append('{}'.format(file_))
        picked.append(class_)
    
    return subset
  

# dataset_file = '/opt/caffe/ristretto/examples/mnist/mnist_test_lmdb/data.mdb'
