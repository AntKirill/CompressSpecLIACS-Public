import os
import sys
from scipy import stats
import numpy as np


def main(experiment_root):
    needed = []
    for item in os.listdir(experiment_root):
        fullpath = os.path.join(experiment_root, item)
        if os.path.isfile(fullpath) and item.startswith('group_samples_'):
            needed.append(fullpath)
    M = np.zeros((len(needed), len(needed)))
    for i, fullpath in enumerate(needed):
        with open(fullpath, 'r') as file:
            lines = file.readlines()
            sp = lines[0].split(' ')
            parent = [int(k) for k in sp[1:641]]
            parent_samples = np.array([float(i) for i in sp[642:]])
            for j in range(1, len(lines)):
                sp = lines[j].split(' ')
                x = [int(i) for i in sp[1:641]]
                x_samples = np.array([float(i) for i in sp[642:]])
                M[i][j-1] = stats.ttest_ind(parent_samples**2, x_samples**2, equal_var=False).pvalue
    with open(os.path.join(experiment_root, 'matrix.csv'), 'w') as file:
        for i in range(len(M)):
            print(*M[i], file=file) 


if __name__ == '__main__':
    main(*sys.argv[1:])
