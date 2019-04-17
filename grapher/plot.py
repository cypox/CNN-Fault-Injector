import networkx as nx
import matplotlib.pyplot as plt

'''
from google.protobuf import text_format
from caffe.proto import caffe_pb2

net_params = caffe_pb2.NetParameter()
text_format.Merge(open(self.prototxt).read(), net_params)
#code for nodes
for l in net_params.layer:
  print("'{}', ".format(str(l.bottom[0]))),

# code for edges
for l in net_params.layer:
  for b in l.bottom:
    if b != l.top[0]:
      print("g.add_edge('{}', '{}')".format(str(b), str(l.top[0])))
'''

g = nx.DiGraph()

nodes = ['data',  'conv1/7x7_s2',  'conv1/7x7_s2',  'pool1/3x3_s2',  'pool1/norm1',  'conv2/3x3_reduce',  'conv2/3x3_reduce',  'conv2/3x3',  'conv2/3x3',  'conv2/norm2',  'pool2/3x3_s2',  'inception_3a/1x1',  'pool2/3x3_s2',  'inception_3a/3x3_reduce',  'inception_3a/3x3_reduce',  'inception_3a/3x3',  'pool2/3x3_s2',  'inception_3a/5x5_reduce',  'inception_3a/5x5_reduce',  'inception_3a/5x5',  'pool2/3x3_s2',  'inception_3a/pool',  'inception_3a/pool_proj',  'inception_3a/1x1',  'inception_3a/output',  'inception_3b/1x1',  'inception_3a/output',  'inception_3b/3x3_reduce',  'inception_3b/3x3_reduce',  'inception_3b/3x3',  'inception_3a/output',  'inception_3b/5x5_reduce',  'inception_3b/5x5_reduce',  'inception_3b/5x5',  'inception_3a/output',  'inception_3b/pool',  'inception_3b/pool_proj',  'inception_3b/1x1',  'inception_3b/output',  'pool3/3x3_s2',  'inception_4a/1x1',  'pool3/3x3_s2',  'inception_4a/3x3_reduce',  'inception_4a/3x3_reduce',  'inception_4a/3x3',  'pool3/3x3_s2',  'inception_4a/5x5_reduce',  'inception_4a/5x5_reduce',  'inception_4a/5x5',  'pool3/3x3_s2',  'inception_4a/pool',  'inception_4a/pool_proj',  'inception_4a/1x1',  'inception_4a/output',  'inception_4b/1x1',  'inception_4a/output',  'inception_4b/3x3_reduce',  'inception_4b/3x3_reduce',  'inception_4b/3x3',  'inception_4a/output',  'inception_4b/5x5_reduce',  'inception_4b/5x5_reduce',  'inception_4b/5x5',  'inception_4a/output',  'inception_4b/pool',  'inception_4b/pool_proj',  'inception_4b/1x1',  'inception_4b/output',  'inception_4c/1x1',  'inception_4b/output',  'inception_4c/3x3_reduce',  'inception_4c/3x3_reduce',  'inception_4c/3x3',  'inception_4b/output',  'inception_4c/5x5_reduce',  'inception_4c/5x5_reduce',  'inception_4c/5x5',  'inception_4b/output',  'inception_4c/pool',  'inception_4c/pool_proj',  'inception_4c/1x1',  'inception_4c/output',  'inception_4d/1x1',  'inception_4c/output',  'inception_4d/3x3_reduce',  'inception_4d/3x3_reduce',  'inception_4d/3x3',  'inception_4c/output',  'inception_4d/5x5_reduce',  'inception_4d/5x5_reduce',  'inception_4d/5x5',  'inception_4c/output',  'inception_4d/pool',  'inception_4d/pool_proj',  'inception_4d/1x1',  'inception_4d/output',  'inception_4e/1x1',  'inception_4d/output',  'inception_4e/3x3_reduce',  'inception_4e/3x3_reduce',  'inception_4e/3x3',  'inception_4d/output',  'inception_4e/5x5_reduce',  'inception_4e/5x5_reduce',  'inception_4e/5x5',  'inception_4d/output',  'inception_4e/pool',  'inception_4e/pool_proj',  'inception_4e/1x1',  'inception_4e/output',  'pool4/3x3_s2',  'inception_5a/1x1',  'pool4/3x3_s2',  'inception_5a/3x3_reduce',  'inception_5a/3x3_reduce',  'inception_5a/3x3',  'pool4/3x3_s2',  'inception_5a/5x5_reduce',  'inception_5a/5x5_reduce',  'inception_5a/5x5',  'pool4/3x3_s2',  'inception_5a/pool',  'inception_5a/pool_proj',  'inception_5a/1x1',  'inception_5a/output',  'inception_5b/1x1',  'inception_5a/output',  'inception_5b/3x3_reduce',  'inception_5b/3x3_reduce',  'inception_5b/3x3',  'inception_5a/output',  'inception_5b/5x5_reduce',  'inception_5b/5x5_reduce',  'inception_5b/5x5',  'inception_5a/output',  'inception_5b/pool',  'inception_5b/pool_proj',  'inception_5b/1x1',  'inception_5b/output',  'pool5/7x7_s1',  'pool5/7x7_s1',  'loss3/classifier', 'prob']
# g.add_nodes_from(nodes)

positions = []
level = 0
ranks = ['3a', '3b', '4a', '4b', '4c', '4d', '4e', '5a', '5b']
for node in nodes:
  x = 0

  if 'inception' not in node:
    x = 0
    level += 1
    y = level
  else:
    key = node.split('/')[1]
    rank = node.split('/')[0].split('_')[1]
    base = level + ranks.index(rank) * 3
    if key == 'output':
      x = 0
      y = base + 3
    elif key == '3x3_reduce':
      x = -1
      y = base + 1
    elif key == '5x5_reduce':
      x = 1
      y = base + 1
    elif key == 'pool':
      x = 2
      y = base + 1
    elif key == '1x1':
      x = -2
      y = base + 2
    elif key == '3x3':
      x = -1
      y = base + 2
    elif key == '5x5':
      x = 1
      y = base + 2
    elif key == 'pool_proj':
      x = 2
      y = base + 2

  positions.append((x, y))

for n, pos in zip(nodes, positions):
  g.add_node(n, pos=pos)

pos = nx.get_node_attributes(g, 'pos')


g.add_edge('data', 'conv1/7x7_s2')
g.add_edge('conv1/7x7_s2', 'pool1/3x3_s2')
g.add_edge('pool1/3x3_s2', 'pool1/norm1')
g.add_edge('pool1/norm1', 'conv2/3x3_reduce')
g.add_edge('conv2/3x3_reduce', 'conv2/3x3')
g.add_edge('conv2/3x3', 'conv2/norm2')
g.add_edge('conv2/norm2', 'pool2/3x3_s2')
g.add_edge('pool2/3x3_s2', 'inception_3a/1x1')
g.add_edge('pool2/3x3_s2', 'inception_3a/3x3_reduce')
g.add_edge('inception_3a/3x3_reduce', 'inception_3a/3x3')
g.add_edge('pool2/3x3_s2', 'inception_3a/5x5_reduce')
g.add_edge('inception_3a/5x5_reduce', 'inception_3a/5x5')
g.add_edge('pool2/3x3_s2', 'inception_3a/pool')
g.add_edge('inception_3a/pool', 'inception_3a/pool_proj')
g.add_edge('inception_3a/1x1', 'inception_3a/output')
g.add_edge('inception_3a/3x3', 'inception_3a/output')
g.add_edge('inception_3a/5x5', 'inception_3a/output')
g.add_edge('inception_3a/pool_proj', 'inception_3a/output')
g.add_edge('inception_3a/output', 'inception_3b/1x1')
g.add_edge('inception_3a/output', 'inception_3b/3x3_reduce')
g.add_edge('inception_3b/3x3_reduce', 'inception_3b/3x3')
g.add_edge('inception_3a/output', 'inception_3b/5x5_reduce')
g.add_edge('inception_3b/5x5_reduce', 'inception_3b/5x5')
g.add_edge('inception_3a/output', 'inception_3b/pool')
g.add_edge('inception_3b/pool', 'inception_3b/pool_proj')
g.add_edge('inception_3b/1x1', 'inception_3b/output')
g.add_edge('inception_3b/3x3', 'inception_3b/output')
g.add_edge('inception_3b/5x5', 'inception_3b/output')
g.add_edge('inception_3b/pool_proj', 'inception_3b/output')
g.add_edge('inception_3b/output', 'pool3/3x3_s2')
g.add_edge('pool3/3x3_s2', 'inception_4a/1x1')
g.add_edge('pool3/3x3_s2', 'inception_4a/3x3_reduce')
g.add_edge('inception_4a/3x3_reduce', 'inception_4a/3x3')
g.add_edge('pool3/3x3_s2', 'inception_4a/5x5_reduce')
g.add_edge('inception_4a/5x5_reduce', 'inception_4a/5x5')
g.add_edge('pool3/3x3_s2', 'inception_4a/pool')
g.add_edge('inception_4a/pool', 'inception_4a/pool_proj')
g.add_edge('inception_4a/1x1', 'inception_4a/output')
g.add_edge('inception_4a/3x3', 'inception_4a/output')
g.add_edge('inception_4a/5x5', 'inception_4a/output')
g.add_edge('inception_4a/pool_proj', 'inception_4a/output')
g.add_edge('inception_4a/output', 'inception_4b/1x1')
g.add_edge('inception_4a/output', 'inception_4b/3x3_reduce')
g.add_edge('inception_4b/3x3_reduce', 'inception_4b/3x3')
g.add_edge('inception_4a/output', 'inception_4b/5x5_reduce')
g.add_edge('inception_4b/5x5_reduce', 'inception_4b/5x5')
g.add_edge('inception_4a/output', 'inception_4b/pool')
g.add_edge('inception_4b/pool', 'inception_4b/pool_proj')
g.add_edge('inception_4b/1x1', 'inception_4b/output')
g.add_edge('inception_4b/3x3', 'inception_4b/output')
g.add_edge('inception_4b/5x5', 'inception_4b/output')
g.add_edge('inception_4b/pool_proj', 'inception_4b/output')
g.add_edge('inception_4b/output', 'inception_4c/1x1')
g.add_edge('inception_4b/output', 'inception_4c/3x3_reduce')
g.add_edge('inception_4c/3x3_reduce', 'inception_4c/3x3')
g.add_edge('inception_4b/output', 'inception_4c/5x5_reduce')
g.add_edge('inception_4c/5x5_reduce', 'inception_4c/5x5')
g.add_edge('inception_4b/output', 'inception_4c/pool')
g.add_edge('inception_4c/pool', 'inception_4c/pool_proj')
g.add_edge('inception_4c/1x1', 'inception_4c/output')
g.add_edge('inception_4c/3x3', 'inception_4c/output')
g.add_edge('inception_4c/5x5', 'inception_4c/output')
g.add_edge('inception_4c/pool_proj', 'inception_4c/output')
g.add_edge('inception_4c/output', 'inception_4d/1x1')
g.add_edge('inception_4c/output', 'inception_4d/3x3_reduce')
g.add_edge('inception_4d/3x3_reduce', 'inception_4d/3x3')
g.add_edge('inception_4c/output', 'inception_4d/5x5_reduce')
g.add_edge('inception_4d/5x5_reduce', 'inception_4d/5x5')
g.add_edge('inception_4c/output', 'inception_4d/pool')
g.add_edge('inception_4d/pool', 'inception_4d/pool_proj')
g.add_edge('inception_4d/1x1', 'inception_4d/output')
g.add_edge('inception_4d/3x3', 'inception_4d/output')
g.add_edge('inception_4d/5x5', 'inception_4d/output')
g.add_edge('inception_4d/pool_proj', 'inception_4d/output')
g.add_edge('inception_4d/output', 'inception_4e/1x1')
g.add_edge('inception_4d/output', 'inception_4e/3x3_reduce')
g.add_edge('inception_4e/3x3_reduce', 'inception_4e/3x3')
g.add_edge('inception_4d/output', 'inception_4e/5x5_reduce')
g.add_edge('inception_4e/5x5_reduce', 'inception_4e/5x5')
g.add_edge('inception_4d/output', 'inception_4e/pool')
g.add_edge('inception_4e/pool', 'inception_4e/pool_proj')
g.add_edge('inception_4e/1x1', 'inception_4e/output')
g.add_edge('inception_4e/3x3', 'inception_4e/output')
g.add_edge('inception_4e/5x5', 'inception_4e/output')
g.add_edge('inception_4e/pool_proj', 'inception_4e/output')
g.add_edge('inception_4e/output', 'pool4/3x3_s2')
g.add_edge('pool4/3x3_s2', 'inception_5a/1x1')
g.add_edge('pool4/3x3_s2', 'inception_5a/3x3_reduce')
g.add_edge('inception_5a/3x3_reduce', 'inception_5a/3x3')
g.add_edge('pool4/3x3_s2', 'inception_5a/5x5_reduce')
g.add_edge('inception_5a/5x5_reduce', 'inception_5a/5x5')
g.add_edge('pool4/3x3_s2', 'inception_5a/pool')
g.add_edge('inception_5a/pool', 'inception_5a/pool_proj')
g.add_edge('inception_5a/1x1', 'inception_5a/output')
g.add_edge('inception_5a/3x3', 'inception_5a/output')
g.add_edge('inception_5a/5x5', 'inception_5a/output')
g.add_edge('inception_5a/pool_proj', 'inception_5a/output')
g.add_edge('inception_5a/output', 'inception_5b/1x1')
g.add_edge('inception_5a/output', 'inception_5b/3x3_reduce')
g.add_edge('inception_5b/3x3_reduce', 'inception_5b/3x3')
g.add_edge('inception_5a/output', 'inception_5b/5x5_reduce')
g.add_edge('inception_5b/5x5_reduce', 'inception_5b/5x5')
g.add_edge('inception_5a/output', 'inception_5b/pool')
g.add_edge('inception_5b/pool', 'inception_5b/pool_proj')
g.add_edge('inception_5b/1x1', 'inception_5b/output')
g.add_edge('inception_5b/3x3', 'inception_5b/output')
g.add_edge('inception_5b/5x5', 'inception_5b/output')
g.add_edge('inception_5b/pool_proj', 'inception_5b/output')
g.add_edge('inception_5b/output', 'pool5/7x7_s1')
g.add_edge('pool5/7x7_s1', 'loss3/classifier')
g.add_edge('loss3/classifier', 'prob')


plt.figure(3,figsize=(3,24))

plt.ylim(30, 0)
plt.xlim(-10, 10)

# nx.draw(g, pos, with_labels=True)
# nx.draw(g, with_labels = True)

nx.draw_networkx_nodes(g, pos, node_size=200, node_color='b', alpha=1.0, node_shape='o') # ‘so^>v<dph8’
nx.draw_networkx_edges(g, pos, alpha=1.0, node_size=200, width=1, edge_color='k')
nx.draw_networkx_labels(g, pos, font_size=12, font_family='sans-serif')

plt.draw()

plt.show()
