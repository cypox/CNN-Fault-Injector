import os, sys
os.environ["GLOG_minloglevel"] = "2"

import caffe
import numpy as np

from google.protobuf import text_format
from caffe.proto import caffe_pb2


class classifier:
  def __init__(self, prototxt, caffemodel, mean, mode):
    self.prototxt = prototxt
    self.caffemodel = caffemodel
    self.mean = mean

    if mode == 'gpu':
      print('[\033[92mmodel   \033[0m]: using gpu for inference.')
      caffe.set_mode_gpu()
      caffe.set_device(0)
    elif mode == 'cpu':
      print('[\033[92mmodel   \033[0m]: using cpu for inference.')
      caffe.set_mode_cpu()
    else:
      print('[\033[92mmodel   \033[0m]: no mode selected. defaulting to cpu')
      caffe.set_mode_cpu()

    self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

    self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
    self.transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
    self.transformer.set_transpose('data', (2,0,1))
    self.transformer.set_channel_swap('data', (2,1,0))
    self.transformer.set_raw_scale('data', 255.0)
  
  def reset(self, prototxt, caffemodel):
    self.prototxt = prototxt
    self.caffemodel = caffemodel
    self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
  
  def predict(self, img):
    #self.net.blobs['data'].reshape(1,3,224,224)
    im = caffe.io.load_image(img)
    self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
    out = self.net.forward()
    return out
  
  def forward_from_to(self, img, from_, to_):
    im = caffe.io.load_image(img)
    self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
    out = self.net.forward(start=from_, end=to_)
    return out

  def get_net(self):
    return self.net
  
  def get_layer_name(self, index):
    return self.net.params.items()[index][0]
  
  def get_param_count(self):
    return len(self.net.params)
  
  def get_blob_count(self):
    return len(self.net.blobs)
  
  def get_net_params(self):
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(self.prototxt).read(), net_params)
    return net_params
  
  def get_total_params(self):
    return np.sum([np.prod(v[0].data.shape) for k, v in self.net.params.items()]) + np.sum([np.prod(v[1].data.shape) for k, v in self.net.params.items()])
