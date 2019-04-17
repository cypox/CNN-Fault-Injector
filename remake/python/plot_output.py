import sys
import re

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.interpolate import spline
from scipy.interpolate import interp1d

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def squeeze_series(values):
  squeezed = np.zeros((10))
  # first conv layer
  squeezed[0] = values[0]
  # next 8 fire modules
  for i in range(0, 8):
    s = i * 3 + 1
    averaged = values[s] + values[s+1] + values[s+2]
    squeezed[i+1] = averaged / 3.
  # last conv layer
  squeezed[9] = values[-1]
  return squeezed

def incept_series(values):
  inception = np.zeros((13))
  # 3 first layers
  for i in range(3):
    inception[i] = values[i]
  # 9 inception modules
  for i in range(0, 9):
    s = i * 6 + 3
    averaged = values[s] + values[s+1] + values[s+2] + values[s+3] + values[s+4] + values[s+5]
    inception[i+3] = averaged / 6.
  inception[12] = values[-1]
  return inception

def plot_mean_multiple(model_name, x, mean_series, std_series, outfile = None, smoothing = True, bar_chart = None):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  plt.rc('font', size=21)
  plt.figure(figsize=(10, 5))
  plt.xlabel('Number of injected errors')
  plt.ylabel('Accuracy (correct predictions)')
  plt.title('{}'.format(model_name.capitalize()))
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
    interp_type = 'linear'
    mean_interp = interp1d(x, mean_series, kind=interp_type)
    mean_smooth = mean_interp(xnew)
    std_interp = interp1d(x, std_series, kind=interp_type)
    std_smooth = std_interp(xnew)
  else:
    xnew = x
    mean_smooth = mean_series
    std_smooth = std_series

  if bar_chart is None:
    plt.errorbar(xnew, mean_smooth[0]/1000, yerr=std_smooth[0]/1000, ls='--', color='red', label=r'$\mathcal{Q}$ accuracy distribution ($\mu$, $\sigma$)'
              , elinewidth=.7, markeredgewidth=.7, capsize=2
              , fmt='o', ecolor='darksalmon', markersize=5)
    plt.errorbar(xnew, mean_smooth[1]/1000, yerr=std_smooth[1]/1000, ls='--', color='blue', label=r'$\mathcal{F}$ Accuracy distribution ($\mu$, $\sigma$)'
              , elinewidth=.7, markeredgewidth=.7, capsize=2
              , fmt='o', ecolor='skyblue', markersize=5)
  else:
    plt.bar(xnew, mean_smooth[0], yerr=std_smooth[0], ls='--', color='red', label=r'$\mathcal{Q}$ accuracy distribution ($\mu$, $\sigma$)'
              , capsize=2 , ecolor='darksalmon')
    plt.bar(xnew, mean_smooth[1], yerr=std_smooth[1], ls='--', color='blue', label=r'$\mathcal{F}$ Accuracy distribution ($\mu$, $\sigma$)'
              , capsize=2 , ecolor='skyblue')
  # plt.plot(x, min_series, color='gray', ls=':')
  # plt.plot(x, max_series, color='gray', ls=':')

  plt.legend()

  plt.xlim(left=0)
  plt.ylim(0, 1.)

  if outfile is not None:
    plt.savefig(outfile, bbox_inches='tight')
  plt.show()

def plot_mean_index(prefix, x, index_series, outfile = None):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  plt.rc('font', size=21)

  x = [l+1 for l in x]
  x.reverse()

  fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=(10, 5))

  fig.text(0.5, 0.0, 'Bit position', ha='center')
  fig.text(0.04, 0.5, 'Accuracy (correct predictions)', va='center', rotation='vertical')

  bw = 32

  plt.xlim(bw+1, 0)
  plt.xticks(range(1, bw+1), rotation=90, ha='center')

  colors = ['red', 'blue', 'green', 'yellow']
  for i in range(len(prefix)):
    ax[i].bar(x, index_series[i]/1000, label=prefix[i].split('/')[1].split('-')[0].capitalize(), color=colors[i])
    for l, j in zip(ax[i].xaxis.get_ticklabels(), range(bw)):
      l.set_visible(True)
      # l.set_fontsize(14)
      if j < 23:
        l.set_color('darkred')
      elif j < 31:
        l.set_color('blue')
      elif j < 32:
        l.set_color('green')

  plt.figlegend()

  if outfile is not None:
    plt.savefig(outfile, bbox_inches='tight')
  plt.show()

def plot_layer_mean_std(model_name, x, mean_series, std_series, min_series, max_series, outfile = None, smoothing = True, bar_chart = None, str_x = None):
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  plt.rc('font', size=21)
  plt.figure(figsize=(10, 5))
  plt.xlabel('Layer')
  plt.ylabel('Accuracy (correct predictions)')
  plt.title('{}'.format(model_name.capitalize()))
  plt.axhline(y=0.1, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=0.2, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=0.3, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=0.4, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=0.5, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=0.6, color='lightgrey', linewidth=1, linestyle='--', zorder=1)
  plt.axhline(y=0.7, color='lightgrey', linewidth=1, linestyle='--', zorder=1)

  xnew = [s.replace('_', '-') for s in x]
  mean_smooth = mean_series
  std_smooth = std_series
  min_smooth = min_series
  max_smooth = max_series

  if model_name == 'squeezenet':
    new_x = []
    new_x.append(xnew[0])
    for i in range(2, 10):
      new_x.append('fire{}'.format(i))
    new_x.append(xnew[-1])
    xnew = new_x
    mean_smooth = squeeze_series(mean_smooth)
    std_smooth = squeeze_series(std_smooth)
    min_series = squeeze_series(min_series)
    max_series = squeeze_series(max_series)
  elif model_name == 'googlenet':
    for l,v in zip(xnew, (mean_smooth/1000)):
      print('{}\t{}'.format(l, v))
    new_x = []
    for i in range(3):
      new_x.append(xnew[i].split('/')[1])
    for i in range(0, 9):
      s = i * 6 + 3
      v = 'i-' + xnew[s].split('/')[0].split('-')[1]
      new_x.append(v)
    new_x.append(xnew[-1].split('/')[0])
    xnew = new_x
    mean_smooth = incept_series(mean_smooth)
    std_smooth = incept_series(std_smooth)
    min_series = incept_series(min_series)
    max_series = incept_series(max_series)

  plt.fill_between(xnew, min_series/1000, max_series/1000, facecolor='lightgray', alpha=0.5, label='Extrema')
  if bar_chart is None:
    plt.errorbar(xnew, mean_smooth/1000, yerr=std_smooth/1000, ls='--', color='black', label=r'Accuracy distribution ($\mu$, $\sigma$)'
              , elinewidth=.7, markeredgewidth=.7, capsize=2
              , fmt='o', ecolor='red', markersize=5)
  else:
    plt.bar(xnew, mean_smooth, yerr=std_smooth, ls='--', color='black', label=r'Accuracy distribution ($\mu$, $\sigma$)'
              , capsize=2 , ecolor='red')
  plt.plot(xnew, min_series/1000, color='gray', ls=':')
  plt.plot(xnew, max_series/1000, color='gray', ls=':')

  plt.legend()
  plt.tight_layout()

  plt.xlim(left=0)
  plt.ylim(0, 0.8)
  #plt.yscale('log')
  #plt.xticks(rotation=45, ha='right', fontsize=10) # ha: x labels start from the tick and are not centered ==> no overlap if not the same size
  plt.xticks(rotation=45, ha='right')

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

def parse_file_layer(input_file, xtype=int):
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
    x.append(xtype(l.split('-')[0]))
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

def plot_file(model_name, prefix, file_type, serie, metric, tests = 30, smooth_graphs = True, outfile = None):
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
    if metric == 5:
      q_channel = 0
      f_channel = 2
    elif metric == 1:
      q_channel = 1
      f_channel = 3
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
    # min_series = np.min(series, axis=0)
    # max_series = np.max(series, axis=0)
    mean_series = [mean_series[q_channel], mean_series[f_channel]]
    std_series = [std_series[q_channel], std_series[f_channel]]
    plot_mean_multiple(model_name, x, mean_series, std_series, outfile = outfile, smoothing=smooth_graphs, bar_chart = None)
  elif file_type == 'index':
    xq = range(8)
    xf = range(32)
    index_series = []
    index_series_q = np.zeros((len(prefix), 8))
    index_series_f = np.zeros((len(prefix), 32))
    for p in range(len(prefix)):
      network = prefix[p]
      q_series = np.zeros((tests, 2, 8))
      f_series = np.zeros((tests, 2, 32))
      for i in range(tests):
        filename = '{}-{:0>3}'.format(network, i)
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
      if channel == 0 or channel == 1:
        x = xq
        mean_series = q_mean_series
        std_series = q_std_series
        min_series = q_min_series
        max_series = q_max_series
        index_series_q[p,:] = mean_series[channel]
      elif channel == 2 or channel == 3:
        x = xf
        mean_series = f_mean_series
        std_series = f_std_series
        min_series = f_min_series
        max_series = f_max_series
        index_series_f[p,:] = mean_series[channel - 2]
    
    if channel == 0 or channel == 1:
      x = xq
      index_series = index_series_q
    elif channel == 2 or channel == 3:
      x = xf
      index_series = index_series_f
    plot_mean_index(prefix, x, index_series, outfile = outfile)
  elif file_type == 'layer':
    x = []
    (x, q_list, f_list) = parse_file_layer('{}-{:0>3}'.format(prefix, 0), xtype=str)
    series = np.zeros((tests, 4, len(q_list[0])))
    for i in range(tests):
      filename = '{}-{:0>3}'.format(prefix, i)
      (x, q_list, f_list) = parse_file_layer(filename, xtype=str)
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
    plot_layer_mean_std(model_name, x, mean_series[channel], std_series[channel], min_series[channel], max_series[channel], outfile = outfile, smoothing = smooth_graphs, bar_chart = None, str_x = True)

model_name = sys.argv[1]
plot_type = sys.argv[2]
quant_type = 'f'
metric_type = 1
total_runs = 60
errors = 10
outfile = 'figures/{}-{}-{}.pdf'.format(model_name, plot_type, metric_type)
if sys.argv[2] == 'layer':
  prefix = 'outputs/{}-{}-{}-{}e/run'.format(model_name, plot_type, total_runs, errors)
elif sys.argv[2] == 'index':
  prefix = ['outputs/{}-{}-{}-{}e/run'.format(M, plot_type, total_runs, errors) for M in ['alexnet', 'vgg16', 'googlenet', 'squeezenet']]
elif sys.argv[2] == 'full':
  prefix = 'outputs/{}-{}-{}/run'.format(model_name, plot_type, total_runs)
plot_file(model_name, prefix, plot_type, quant_type, metric_type, tests=total_runs, smooth_graphs=False, outfile=outfile)
