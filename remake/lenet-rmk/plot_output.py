import sys
import re

import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.interpolate import spline
from scipy.interpolate import interp1d

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_mean_std(x, mean_series, std_series, min_series, max_series, outfile = None, smoothing = True, bar_chart = None):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  plt.rc('font', size=21)
  plt.figure(figsize=(10, 5))
  plt.xlabel('\\# of layer')
  plt.ylabel('Accuracy (correct predictions)')
  plt.title('Impact of cumulative memory errors on accuracy')
  plt.axhline(y=100, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=200, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=300, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=400, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=500, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=600, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=700, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=800, color='lightgrey', linewidth=1, linestyle='--', zorder=1)

  rline = mlines.Line2D([], [], color='red', markersize=15, label='1 bit-flip')
  gline = mlines.Line2D([], [], color='green', markersize=15, label='2 bit-flips')
  bline = mlines.Line2D([], [], color='black', markersize=15, label='3 bit-flips')
  plt.legend(handles=[rline, gline, bline])

  if smoothing == True:
    xnew = np.linspace(np.min(x), np.max(x), 100, endpoint=True)
    order = 3
    interp_type = 'cubic'
    #mean_smooth = spline(x, mean_series, xnew, order=order)
    #std_smooth = spline(x, std_series, xnew, order=order)
    #min_smooth = spline(x, min_series, xnew, order=order)
    #max_smooth = spline(x, max_series, xnew, order=order)
    mean_interp = interp1d(x, mean_series, kind=interp_type)
    mean_smooth = mean_interp(xnew)
    std_interp = interp1d(x, std_series, kind=interp_type)
    std_smooth = std_interp(xnew)
    min_interp = interp1d(x, min_series, kind=interp_type)
    min_smooth = min_interp(xnew)
    max_interp = interp1d(x, max_series, kind=interp_type)
    max_smooth = max_interp(xnew)
  else:
    xnew = x
    mean_smooth = mean_series
    std_smooth = std_series
    min_smooth = min_series
    max_smooth = max_series

  plt.fill_between(x, min_series, max_series, facecolor='lightgray', alpha=0.5, label='Extrema')
  if bar_chart is None:
    plt.errorbar(xnew, mean_smooth, yerr=std_smooth, ls='--', color='black', label=r'Accuracy distribution ($\mu$, $\sigma$)'
              , elinewidth=.7, markeredgewidth=.7, capsize=2
              , fmt='o', ecolor='red', markersize=5)
  else:
    plt.bar(xnew, mean_smooth, yerr=std_smooth, ls='--', color='black', label=r'Accuracy distribution ($\mu$, $\sigma$)'
              , capsize=2 , ecolor='red')
  plt.plot(x, min_series, color='gray', ls=':')
  plt.plot(x, max_series, color='gray', ls=':')

  plt.legend()

  if outfile is not None:
    plt.savefig(outfile, bbox_inches='tight')
  plt.show()

def parse_file_full(input_file):
  f = open(input_file, 'r')
  lines = f.readlines()
  f.close()
  total = len(lines)
  q_lines = [x.replace('\n', '') for x in lines[1:total//2]]
  f_lines = [x.replace('\n', '') for x in lines[total//2 + 1:]]
  x = []
  t1 = []
  t5 = []
  for l in q_lines:
    x.append(int(l.split('-')[0]))
    t5.append(int(l.split('-')[1]))
    t1.append(int(l.split('-')[2]))
  q_list = (t5, t1)
  t1 = []
  t5 = []
  for l in f_lines:
    t5.append(int(l.split('-')[1]))
    t1.append(int(l.split('-')[2]))
  f_list = (t5, t1)
  return (x, q_list, f_list)

def parse_file_index(input_file):
  f = open(input_file, 'r')
  lines = f.readlines()
  f.close()
  total = len(lines)
  q_lines = [x.replace('\n', '') for x in lines[1:9]]
  f_lines = [x.replace('\n', '') for x in lines[10:]]
  t1 = []
  t5 = []
  for l in q_lines:
    t5.append(int(l.split('-')[1]))
    t1.append(int(l.split('-')[2]))
  q_list = (t5, t1)
  t1 = []
  t5 = []
  for l in f_lines:
    t5.append(int(l.split('-')[1]))
    t1.append(int(l.split('-')[2]))
  f_list = (t5, t1)
  return (q_list, f_list)

def parse_file_layer(input_file):
  f = open(input_file, 'r')
  lines = f.readlines()
  f.close()
  total = len(lines)
  q_lines = [x.replace('\n', '') for x in lines[1:total//2]]
  f_lines = [x.replace('\n', '') for x in lines[total//2 + 1:]]
  x = []
  t1 = []
  t5 = []
  for l in q_lines:
    x.append(int(l.split('-')[0]))
    t5.append(int(l.split('-')[1]))
    t1.append(int(l.split('-')[2]))
  q_list = (t5, t1)
  t1 = []
  t5 = []
  for l in f_lines:
    t5.append(int(l.split('-')[1]))
    t1.append(int(l.split('-')[2]))
  f_list = (t5, t1)
  return (x, q_list, f_list)

def parse_file(prefix, file_type, serie, metric, tests = 30, smooth_graphs = True, outfile = None):
  if serie == 'q' and metric == 5:
    channel = 0
  elif serie == 'q' and metric == 1:
    channel = 1
  elif serie == 'f' and metric == 5:
    channel = 2
  elif serie == 'f' and metric == 1:
    channel = 3
  # channel =: 0 for top 5 quantized, 1 for top 1 quantized, 2 for top 5 float, 3 for top 1 float
  if file_type == 'full':
    x = []
    series = np.zeros((tests, 4, 14))
    for i in range(tests):
      filename = '{}-{:0>3}'.format(prefix, i)
      (x, q_list, f_list) = parse_file_full(filename)
      series[i,0,:] = q_list[0]
      series[i,1,:] = q_list[1]
      series[i,2,:] = f_list[0]
      series[i,3,:] = f_list[1]
    mean_series = np.mean(series, axis=0)
    std_series = pow(np.std(series, axis=0), 1)
    min_series = np.min(series, axis=0)
    max_series = np.max(series, axis=0)
    # print(series[:,2,0], pow(np.std(series[:,2,0]), 2))
    # print(std_series[2])
    plot_mean_std(x, mean_series[channel], std_series[channel], min_series[channel], max_series[channel], outfile = outfile, smoothing=smooth_graphs, bar_chart = None)
  elif file_type == 'index':
    xq = range(8)
    xf = range(32)
    q_series = np.zeros((tests, 2, 8))
    f_series = np.zeros((tests, 2, 32))
    for i in range(tests):
      filename = '{}-{:0>3}'.format(prefix, i)
      (q_list, f_list) = parse_file_index(filename)
      q_series[i,0,:] = q_list[0]
      q_series[i,1,:] = q_list[1]
      f_series[i,0,:] = f_list[0]
      f_series[i,1,:] = f_list[1]
    q_mean_series = np.mean(q_series, axis=0)
    q_std_series = pow(np.std(q_series, axis=0), 1)
    q_min_series = np.min(q_series, axis=0)
    q_max_series = np.max(q_series, axis=0)
    f_mean_series = np.mean(f_series, axis=0)
    f_std_series = pow(np.std(f_series, axis=0), 1)
    f_min_series = np.min(f_series, axis=0)
    f_max_series = np.max(f_series, axis=0)
    # print(series[:,2,0], pow(np.std(series[:,2,0]), 2))
    # print(std_series[2])
    if channel == 0 or channel == 1:
      mean_series = q_mean_series
      std_series = q_std_series
      min_series = q_min_series
      max_series = q_max_series
    elif channel == 2 or channel == 3:
      channel -= 2
      mean_series = f_mean_series
      std_series = f_std_series
      min_series = f_min_series
      max_series = f_max_series
    plot_mean_std(xf, mean_series[channel], std_series[channel], min_series[channel], max_series[channel], outfile = outfile, smoothing=False, bar_chart = True)
  elif file_type == 'layer':
    x = []
    (x, q_list, f_list) = parse_file_layer('{}-{:0>3}'.format(prefix, 0))
    series = np.zeros((tests, 4, len(q_list[0])))
    for i in range(tests):
      filename = '{}-{:0>3}'.format(prefix, i)
      (x, q_list, f_list) = parse_file_layer(filename)
      series[i,0,:] = q_list[0]
      series[i,1,:] = q_list[1]
      series[i,2,:] = f_list[0]
      series[i,3,:] = f_list[1]
    mean_series = np.mean(series, axis=0)
    std_series = pow(np.std(series, axis=0), 1)
    min_series = np.min(series, axis=0)
    max_series = np.max(series, axis=0)
    # print(series[:,2,0], pow(np.std(series[:,2,0]), 2))
    # print(std_series[2])
    plot_mean_std(x, mean_series[channel], std_series[channel], min_series[channel], max_series[channel], outfile = outfile, smoothing = smooth_graphs, bar_chart = None)

model_name = sys.argv[1]
plot_type = sys.argv[2]
if sys.argv[2] != 'full':
  prefix = 'outputs/{}-{}-30-50e/run'.format(model_name, plot_type)
else:
  prefix = 'outputs/{}-{}-30/run'.format(model_name, plot_type)
parse_file(prefix, plot_type, 'f', 5, tests=30, smooth_graphs=False, outfile=None)
