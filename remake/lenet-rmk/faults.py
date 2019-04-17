from __future__ import division # for floating point division

import caffe
import numpy as np
import struct

class error:
  def __init__(self, layer, position, index_f, index_q, is_bias):
    self.layer = layer
    self.position = position
    self.bit_index_f = index_f
    self.bit_index_q = index_q
    self.is_bias = is_bias
  
  def set_index(self, index):
    self.bit_index_f = index
    self.bit_index_q = index

  def get_layer(self):
    return self.layer
  
  def get_position(self):
    return self.position
  
  def get_index(self, fixed):
    if fixed == 1:
      return self.bit_index_q
    else:
      return self.bit_index_f
  
  def get_is_bias(self):
    return self.is_bias

def trim_to_fp(before, bw, fl):
  # saturate
  max_data = (pow(2, bw - 1) - 1) * pow(2, -fl)
  min_data = -pow(2, bw - 1) * pow(2, -fl)
  saturated = max(min(before, max_data), min_data)
  # round
  saturated /= pow(2, -fl)
  rounded = round(saturated)
  quantized = rounded * pow(2, -fl)
  return quantized

def float_to_bin(num):
  return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

def bin_to_float(str):
  byte1 = chr(int(str[0:8], 2))
  byte2 = chr(int(str[8:16], 2))
  byte3 = chr(int(str[16:24], 2))
  byte4 = chr(int(str[24:], 2))
  return struct.unpack('!f', byte1+byte2+byte3+byte4)[0]

def fixed_to_bin(num, iw, dw):
  num = abs(num)
  integer = int(num)
  integer_bin = bin(integer)[2:]

  integer_bin = '{0:0<{width}}'.format(integer_bin, width=iw)
  if iw == 0:
    integer_bin = ''
  elif len(integer_bin) > iw:
    print('[\033[94minjector\033[0m]: warning. number overflow when converting to binary. {} cannot be converted into {}-bit binary.'.format(integer, iw))
    integer_bin = integer_bin[-iw:]

  decimal = num - integer
  representation = list()
  while decimal > 0:
    mul = decimal * 2
    if int(mul) == 1:
      representation.append('1')
    else:
      representation.append('0')
    decimal = mul - int(mul)
  decimal_bin = ''.join(representation)
  decimal_bin = '{0:0<{width}}'.format(decimal_bin, width=dw)
  if dw == 0:
    decimal_bin = ''
  elif len(decimal_bin) > dw:
    print(decimal_bin)
    print('[\033[94minjector\033[0m]: warning. number overflow when converting to binary. {} cannot be converted into {}-bit binary.'.format(decimal, dw))
    decimal_bin = decimal_bin[:dw]

  # assert(len(integer_bin) == iw)
  # assert(len(decimal_bin) == dw)

  # print('binary of {} is {}.{}'.format(num, integer_bin, decimal_bin))

  return (integer_bin, decimal_bin)

def bin_to_fixed(s, integer, decimal):
  if len(integer) == 0:
    int_part = 0
  else:
    int_part = int(''.join(integer), 2)
  if len(decimal) == 0:
    dec_part = 0.
  else:
    dec_part = 0.
    for i in range(len(decimal)):
      dec_part += int(decimal[i], 2) * pow(2, -(i+1))
  after = s * (int_part + dec_part)
  return after

class injector:
  def __init__(self, model, quantized):
    self.net = model.get_net()
    self.net_params = model.get_net_params()
    self.net_total_params = model.get_total_params()
    if quantized == 'q':
      print('[\033[94minjector\033[0m]: injecting as fixed-point')
      self.fixed = 1
    elif quantized == 'f':
      print('[\033[94minjector\033[0m]: injecting as floating-point')
      self.fixed = 0
    else:
      print('[\033[94minjector\033[0m]: warning. quantization not recognized')
      self.fixed = 2
    
    self.stats = np.zeros(32)
  
  def reset(self, model, quantized):
    self.net = model.get_net()
    self.net_params = model.get_net_params()
    self.net_total_params = model.get_total_params()
    if quantized == 'q':
      print('[\033[94minjector\033[0m]: reset as fixed-point')
      self.fixed = 1
    elif quantized == 'f':
      print('[\033[94minjector\033[0m]: reset as floating-point')
      self.fixed = 0
    else:
      print('[\033[94minjector\033[0m]: warning. quantization not recognized')
      self.fixed = 2

    self.stats = np.zeros(32)
  
  def get_stats(self):
    return self.stats
  
  def generate_random_errors(self, sample_size, layer_index = None, bit_index = None):
    error_list = []
  
    for i in range(sample_size):
      if layer_index is None:
        # choose target layer proportional to number of weights
        injection_index = np.random.randint(0, self.net_total_params)
        last_scanned = 0
        for k, v in self.net.params.items():
          last_scanned += np.prod(v[0].data.shape) + np.prod(v[1].data.shape)
          if last_scanned > injection_index:
            # print('choosed layer {}. {}/{}'.format(k, injection_index, self.net_total_params))
            layer = k
            break
      else:
        layer = self.net.params.keys()[layer_index]
      
      # choose whether to inject in weights or biases
      num_weights = np.prod(self.net.params[layer][0].data.shape)
      num_biases = np.prod(self.net.params[layer][1].data.shape)
      ratio = num_weights / (num_weights + num_biases)
      prob = np.random.rand()
      if prob > num_weights:
        is_bias = 1
      else:
        is_bias = 0

      # select random position from layer
      shape = self.net.params[layer][is_bias].data.shape
      position = ()
      for i in range(len(shape)):
        position += (np.random.randint(shape[i]), )
      
      # select bit index
      if bit_index is None:
        index_f = np.random.randint(0, 32)
        index_q = np.random.randint(0, 8)
      else:
        index_f = bit_index
        index_q = bit_index
      
      E = error(layer, position, index_f, index_q, is_bias)
      error_list.append(E)

    return error_list
  
  def inject_single_error(self, layer, position, is_bias, fixed, fixed_bit_index):
    after = self.flip_random_bit(layer, position, is_bias, fixed, fixed_bit_index)
    self.net.params[layer][is_bias].data[position] = after

  def inject_seu(self, layer_index = None, bit_index = None):
    if layer_index is None:
      # choose target layer proportional to number of weights
      injection_index = np.random.randint(0, self.net_total_params)
      last_scanned = 0
      for k, v in self.net.params.items():
        last_scanned += np.prod(v[0].data.shape) + np.prod(v[1].data.shape)
        if last_scanned > injection_index:
          # print('choosed layer {}. {}/{}'.format(k, injection_index, self.net_total_params))
          layer = k
          break
      self.inject_in_layer(-1, layer_name = layer, bit_index = bit_index)
    else:
      self.inject_in_layer(layer_index, bit_index = bit_index)

  def inject_in_layer(self, layer_index, layer_name = '', bit_index = None):
    # get layer key
    if layer_index == -1 and layer_name != '':
      layer = layer_name
    else:
      layer = self.net.params.keys()[layer_index]
    # choose whether to inject in weights or biases
    num_weights = np.prod(self.net.params[layer][0].data.shape)
    num_biases = np.prod(self.net.params[layer][1].data.shape)
    ratio = num_weights / (num_weights + num_biases)
    prob = np.random.rand()
    if prob > num_weights:
      is_bias = 1
    else:
      is_bias = 0
    # print('layer has {} w and {} b. injecting in bias {} ({})'.format(num_weights, num_biases, is_bias, prob))
    # select random position from layer
    shape = self.net.params[layer][is_bias].data.shape
    position = ()
    for i in range(len(shape)):
      position += (np.random.randint(shape[i]), )
    # inject random bitflip in selected weight
    after = self.flip_random_bit(layer, position, in_bias=is_bias, fixed=self.fixed, fixed_bit_index = bit_index)
    self.net.params[layer][is_bias].data[position] = after

  def flip_random_bit(self, layer, position, in_bias=0, fixed=1, fixed_bit_index=-1):
    before = self.net.params[layer][in_bias].data[position]
    binary_before = ''
    binary_after = ''
    
    # get layer quantization paramsparams[layer][0].data[position]
    after = 0
    if fixed == 1:
      # print('injecting as fixed-point')
      bw = fl = 0
      for l in self.net_params.layer:
        if l.name == layer:
          bw = l.quantization_param.bw_params
          fl = l.quantization_param.fl_params
      iw = bw - fl - 1
      if iw < 0:
        iw = 0
      dw = fl
      if fl <= 0:
        dw = 0
      
      before = trim_to_fp(before, bw, fl)
      quantized = before
      if quantized < 0:
        s = -1
      else:
        s = 1

      # convert to binary
      (integer, decimal) = fixed_to_bin(quantized, iw, dw)
      binary_before = integer + '.' + decimal
      integer = list(integer)
      decimal = list(decimal)
      # print(integer, decimal)

      if fixed_bit_index is None:
        # random bit-flip
        bit_index = np.random.randint(0, bw)
        self.stats[bit_index] += 1
      else:
        bit_index = fixed_bit_index
      # print('injecting at position {}'.format(bit_index))
      if bit_index == bw - 1:
        s = -s
      else:
        if bit_index < iw:
          integer[bit_index] = str(int(not(int(integer[bit_index]))))
        else:
          bit_index -= iw
          decimal[bit_index] = str(int(not(int(decimal[bit_index]))))
      # print(integer, decimal)
      binary_after = ''.join(integer) + '.' + ''.join(decimal)
      after = bin_to_fixed(s, integer, decimal)
    elif fixed == 0:
      # print('injecting as floating-point')
      binary_repr = float_to_bin(before)
      binary_before = binary_repr
      # print(binary_repr)
      if fixed_bit_index is None:
        bit_index = np.random.randint(0, 32)
        self.stats[bit_index] += 1
      else:
        bit_index = fixed_bit_index
      # print('injecting at position {}'.format(bit_index))
      binary_repr_list = list(binary_repr)
      binary_repr_list[bit_index] = str(int(not(int(binary_repr_list[bit_index]))))
      binary_repr = ''.join(binary_repr_list)
      binary_after = binary_repr
      # print(binary_repr)
      after = bin_to_float(binary_repr)
    else:
      print('[\033[94minjector\033[0m]: warning. no quantization specified. using float-point format for injection')
      # print('injecting as floating-point')
      binary_repr = float_to_bin(before)
      binary_before = binary_repr
      # print(binary_repr)
      if fixed_bit_index is None:
        bit_index = np.random.randint(0, 32)
        self.stats[bit_index] += 1
      else:
        bit_index = fixed_bit_index
      self.stats[bit_index] += 1
      # print('injecting at position {}'.format(bit_index))
      binary_repr_list = list(binary_repr)
      binary_repr_list[bit_index] = str(int(not(int(binary_repr_list[bit_index]))))
      binary_repr = ''.join(binary_repr_list)
      binary_after = binary_repr
      # print(binary_repr)
      after = bin_to_float(binary_repr)
    print('[\033[94minjector\033[0m]: layer {} position {} at index {} was {} ({}) became {} ({})'.format(layer, position, bit_index, before, binary_before, after, binary_after))

    return after
