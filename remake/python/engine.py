import os
from random import shuffle
import numpy as np

from model import classifier
from faults import injector
from extractor import dataset


class engine:
  def __init__(self, dataset_folder, mode, mean, label, name, quantized, prefix, subset_size):
    self.dataset = dataset_folder
    self.prefix = prefix

    self.ds = dataset(self.dataset, label)
    self.random_pick = False
    self.max_images = subset_size
    self.filenames = self.ds.get_subset(max_size=self.max_images)

    self.mean = mean
    self.mode = mode

    self.model_file =   prefix + '/' + name + '/' + quantized + '_model.prototxt'
    self.weights_file = prefix + '/' + name + '/' + quantized + '_weights.caffemodel'
    self.quantized = quantized
    self.net = classifier(self.model_file, self.weights_file, self.mean, self.mode)
    self.inj = injector(self.net, self.quantized)
  
  def get_model_layer_count(self):
    return self.net.get_param_count()
  
  def get_proportional_layer_sizes(self):
    # find minimum number of params
    sizes = [np.prod(v[0].data.shape) + np.prod(v[1].data.shape) for k, v in self.net.get_net().params.items()]
    minimum = np.min(sizes)

    # divide by minimum
    ratios = [x/float(minimum) for x in sizes]
    return ratios

  def reset(self, name=None, quantized=None):
    if name is None and quantized is None:
      self.net.reset(self.model_file, self.weights_file)
      self.inj.reset(self.net, self.quantized)
    else:
      self.model_file =   self.prefix + '/' + name + '/' + quantized + '_model.prototxt'
      self.weights_file = self.prefix + '/' + name + '/' + quantized + '_weights.caffemodel'
      self.quantized = quantized
      self.net.reset(self.model_file, self.weights_file)
      self.inj.reset(self.net, self.quantized)
  
  def generate_error_list(self, sample_size, layer_index = None, bit_index = None):
    error_list = self.inj.generate_random_errors(sample_size, layer_index, bit_index)
    # print('[\033[95mengine  \033[0m]: generated {} errors'.format(sample_size))
    for e in error_list:
      print('[\033[95mengine  \033[0m]: generated error (layer: {}, position: {}, f-index: {}, q-index: {}, in bias: {})'.format(e.get_layer(), e.get_position(), e.get_index(0), e.get_index(1), e.get_is_bias()))
    return error_list
  
  def inject_generated_errors(self, error_list, max_injections = None, inject_index = None):
    if self.quantized == 'q':
      fixed = 1
    else:
      fixed = 0
    if max_injections is None:
      max_injections = len(error_list)
    injected = 0
    for e in error_list:
      if injected == max_injections:
        break
      if inject_index is not None:
        e.set_index(inject_index)
      self.inj.inject_single_error(e.get_layer(), e.get_position(), e.get_is_bias(), fixed, e.get_index(fixed))
      injected += 1

  def perform_test_index(self, number_of_errors, flip_position):
    correct_predictions = self.perform_test(number_of_errors, bit_index = flip_position)
    # stats = self.inj.get_stats()

    return correct_predictions

  def perform_test_full(self, number_of_errors):
    correct_predictions = self.perform_test(number_of_errors)

    return correct_predictions
  
  def perform_layer_wise_test(self, number_of_errors):
    number_of_layers = self.net.get_param_count()
    layer_predictions = []
    for l in range(number_of_layers):
      correct_predictions = self.perform_test(number_of_errors, layer_index=l)
      layer_predictions.append(correct_predictions)

    return layer_predictions
  
  def perform_layer_test(self, number_of_errors, layer_index):
    number_of_layers = self.net.get_param_count()
    layer_predictions = []
    correct_predictions = self.perform_test(number_of_errors, layer_index=layer_index)

    return correct_predictions

  def perform_test(self, number_of_errors, layer_index = None, bit_index = None):
    # print('[\033[95mengine  \033[0m]: testing with {} errors on {} images in layer {}'.format(number_of_errors, max_images, layer_index))
    if layer_index is None:
      index_char = 'X'
    else:
      index_char = str(layer_index)
    self.reset()
    for i in range(number_of_errors):
      self.inj.inject_seu(layer_index=layer_index, bit_index = bit_index)

    total_predictions = 0.
    top_5_correct = top_1_correct = 0

    if self.random_pick == True:
      filenames = os.listdir(self.dataset)
      shuffle(filenames)
    else:
      filenames = self.filenames

    for filename in filenames:
      if total_predictions > self.max_images:
        break
      if filename.endswith(".JPEG"):
        img_file = os.path.join(self.dataset, filename)
        out = self.net.predict(img_file)
        total_predictions += 1

        (t5, t1) = self.ds.is_correct_prediction(filename, out['prob'])
        top_5_correct += t5
        top_1_correct += t1
        
        # print('[\033[95mengine  \033[0m]: {}-{:0>3}-{:0>5}|correct predictions: {:.4f}|progress: {:.4f}%\r'.format(self.quantized, index_char, number_of_errors, correct_predictions/total_predictions, total_predictions*100/float(max_images))), # it should be divied by 50k and multiplied by 100 for percent values
        print('[\033[95mengine  \033[0m]: {}-{:0>3}-{:0>5}|top-5: {:.4f}|top-1: {:.4f}|progress: {:.2f}%\r'.format(self.quantized, index_char, number_of_errors, top_5_correct/total_predictions, top_1_correct/total_predictions, total_predictions*100/float(self.max_images))), # it should be divied by 50k and multiplied by 100 for percent values
    print('[\033[95mengine  \033[0m]: {}-{:0>3}-{:0>5}|top-5: {:.4f}|top-1: {:.4f}|progress: {:.2f}%'.format(self.quantized, index_char, number_of_errors, top_5_correct/total_predictions, top_1_correct/total_predictions, total_predictions*100/float(self.max_images))) # it should be divied by 50k and multiplied by 100 for percent values
    return (top_5_correct, top_1_correct)

  def perform_test_no_reset_no_inject(self, number_of_errors, layer_index = None):
    # print('[\033[95mengine  \033[0m]: testing with {} errors on {} images in layer {}'.format(number_of_errors, max_images, layer_index))
    if layer_index is None:
      index_char = 'X'
    else:
      index_char = str(layer_index)

    total_predictions = 0.
    top_5_correct = top_1_correct = 0

    if self.random_pick == True:
      filenames = os.listdir(self.dataset)
      shuffle(filenames)
    else:
      filenames = self.filenames

    for filename in filenames:
      if total_predictions > self.max_images:
        break
      if filename.endswith(".JPEG"):
        img_file = os.path.join(self.dataset, filename)
        out = self.net.predict(img_file)
        total_predictions += 1

        (t5, t1) = self.ds.is_correct_prediction(filename, out['prob'])
        top_5_correct += t5
        top_1_correct += t1
        
        # print('[\033[95mengine  \033[0m]: {}-{:0>3}-{:0>5}|correct predictions: {:.4f}|progress: {:.4f}%\r'.format(self.quantized, index_char, number_of_errors, correct_predictions/total_predictions, total_predictions*100/float(max_images))), # it should be divied by 50k and multiplied by 100 for percent values
        print('[\033[95mengine  \033[0m]: {}-{:0>3}-{:0>5}|top-5: {:.4f}|top-1: {:.4f}|progress: {:.2f}%\r'.format(self.quantized, index_char, number_of_errors, top_5_correct/total_predictions, top_1_correct/total_predictions, total_predictions*100/float(self.max_images))), # it should be divied by 50k and multiplied by 100 for percent values
    print('[\033[95mengine  \033[0m]: {}-{:0>3}-{:0>5}|top-5: {:.4f}|top-1: {:.4f}|progress: {:.2f}%'.format(self.quantized, index_char, number_of_errors, top_5_correct/total_predictions, top_1_correct/total_predictions, total_predictions*100/float(self.max_images))) # it should be divied by 50k and multiplied by 100 for percent values
    return (top_5_correct, top_1_correct)
