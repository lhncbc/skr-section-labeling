import sys

input = sys.argv[1]

with open(input, 'r') as f:
    cur = ''
    count = 0
    for line in f:
        tokens = line.split('|')
        if cur == '':
            cur = tokens[0]
            count = count + 1
        if tokens[0] != cur:
            cur = tokens[0]
            count = count + 1
    print count
