import signal
import sys
import time

from engine import engine

def signal_handler(sig, frame):
  print('\n[\033[1mmain    \033[0m]: got ctrl+c. exitting.')
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def perform_full_test(errors, model_name, e):
  q_result = 'q,f,{}\n'.format(len(errors))
  f_result = 'f,f,{}\n'.format(len(errors))

  error_list = e.generate_error_list(500, layer_index = None, bit_index = None)

  for i in errors:
    e.reset(model_name, 'q')
    e.inject_generated_errors(error_list, max_injections = i)
    start = time.time()
    (t5, t1) = e.perform_test_no_reset_no_inject(i, layer_index = None)
    end = time.time()
    print('[\033[1mmain    \033[0m]: test took {:.2f}s'.format(end - start))
    q_result += '{}-{}-{}\n'.format(i, t5, t1)

  for i in errors:
    e.reset(model_name, 'f')
    e.inject_generated_errors(error_list, max_injections = i)
    start = time.time()
    (t5, t1) = e.perform_test_no_reset_no_inject(i, layer_index = None)
    end = time.time()
    print('[\033[1mmain    \033[0m]: test took {:.2f}s'.format(end - start))
    f_result += '{}-{}-{}\n'.format(i, t5, t1)
  
  return (q_result, f_result)

def perform_index_test(model_name, e):
  q_result = 'q,i,{}\n'.format(8)
  f_result = 'f,i,{}\n'.format(32)

  error_list = e.generate_error_list(50, layer_index = None, bit_index = None)

  for i in range(8):
    e.reset(model_name, 'q')
    e.inject_generated_errors(error_list, max_injections = None, inject_index = i)
    start = time.time()
    (t5, t1) = e.perform_test_no_reset_no_inject(50, layer_index = None)
    end = time.time()
    print('[\033[1mmain    \033[0m]: test took {:.2f}s'.format(end - start))
    q_result += '{}-{}-{}\n'.format(i, t5, t1)

  for i in range(32):
    e.reset(model_name, 'f')
    e.inject_generated_errors(error_list, max_injections = None, inject_index = i)
    start = time.time()
    (t5, t1) = e.perform_test_no_reset_no_inject(50, layer_index = None)
    end = time.time()
    print('[\033[1mmain    \033[0m]: test took {:.2f}s'.format(end - start))
    f_result += '{}-{}-{}\n'.format(i, t5, t1)

  return (q_result, f_result)

def perform_layer_test(min_errors, layers, model_name, e):
  q_result = 'q,l,{}\n'.format(layers)
  f_result = 'f,l,{}\n'.format(layers)

  ratios = e.get_proportional_layer_sizes()

  for i in range(layers):
    layer_name = e.net.get_layer_name(i)
    num_errors = int(min_errors * ratios[i])
    error_list = e.generate_error_list(num_errors, layer_index = i, bit_index = None)
    
    e.reset(model_name, 'q')
    e.inject_generated_errors(error_list, max_injections = None)
    start = time.time()
    (t5, t1) = e.perform_test_no_reset_no_inject(num_errors, layer_index = i)
    end = time.time()
    print('[\033[1mmain    \033[0m]: test took {:.2f}s'.format(end - start))
    q_result += '{}-{}-{}\n'.format(layer_name, t5, t1)

    e.reset(model_name, 'f')
    e.inject_generated_errors(error_list, max_injections = None)
    start = time.time()
    (t5, t1) = e.perform_test_no_reset_no_inject(num_errors, layer_index = i)
    end = time.time()
    print('[\033[1mmain    \033[0m]: test took {:.2f}s'.format(end - start))
    f_result += '{}-{}-{}\n'.format(layer_name, t5, t1)

  return (q_result, f_result)

mode = 'gpu'
dataset = '/home/ssd/ilsvrc/validation/'
mean_file = '/opt/caffe/ristretto/python/caffe/imagenet/ilsvrc_2012_mean.npy'
label_file = '/opt/caffe/ristretto/data/ilsvrc12/val.txt'
prefix = '/home/aiembed/date-iccd-paper/'
quantized = 'q' # 'q' for quantized fixed-point or 'f' for floating-point representation

if len(sys.argv) >= 4:
  model_name = sys.argv[1]
  test_type = sys.argv[2]
  output_folder = sys.argv[3]
else:
  print('[\033[1mmain    \033[0m]: no arguments given. using default parameters. stop? (y or anything else to continue)'),
  ans = raw_input()
  if ans == 'y':
    exit(0)
  model_name = 'googlenet'
  test_type = 'layer'
  output_folder = ''

max_test_images = 1000
if len(sys.argv) == 5:
  max_test_images = int(sys.argv[4])


e = engine(dataset, mode, mean_file, label_file, model_name, quantized, prefix, max_test_images)

errors = [0, 1, 5, 10, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]

sample_size = 10

total_start = time.time()

for iteration in range(60):

  if test_type == 'full':
    (q_result, f_result) = perform_full_test(errors, model_name, e)
  elif test_type == 'index':
    (q_result, f_result) = perform_index_test(model_name, e)
  elif test_type == 'layer':
    min_errors = 1
    (q_result, f_result) = perform_layer_test(min_errors, e.get_model_layer_count(), model_name, e)

  output = open('{}/run-{:0>3}'.format(output_folder, iteration), 'w')
  output.write(q_result)
  output.write(f_result)
  output.close()

total_end = time.time()
print('[\033[1mmain    \033[0m]: total duration of test {:.2f}s'.format(total_end - total_start))
