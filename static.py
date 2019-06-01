import os
import matplotlib.pyplot as plt
import numpy as np

def _parse(line):
    _, gt, pred = line.strip().split(' ')
    return [float(gt), float(pred)]

with open('cosine_score.txt', 'r') as f:
    lines = f.readlines()

lines = np.array(list(map(_parse, lines)))

plt.figure()
plt.scatter(np.arange(lines.shape[0]), lines[:, 1], c=lines[:, 0].astype('int'))
plt.show()
...