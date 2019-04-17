import sys

prefix1 = sys.argv[1]
prefix2 = sys.argv[2]
i = 0

for i in range(30):
  filename = prefix1 + '/run-{:0>3}'.format(i)
  f = open(filename, 'r')
  lines = f.readlines()
  f.close()
  new_file = prefix2 + '/run-{:0>3}'.format(i+30)
  with open(new_file, 'w') as f:
    for line in lines:
      f.write(line)
  print('{} became {}'.format(filename, new_file))

