import sys
import os
import pandas as pd
import numpy as np


def read_data_frame(dir):
    everyevals = []
    for item in os.listdir(dir):
        fullpath = os.path.join(dir, item)
        if os.path.isdir(fullpath) and item.startswith('everyeval'):
            everyevals.append(fullpath)

    dfs = []
    for everyeval in everyevals:
        for root, directories, files in os.walk(everyeval):
            for filename in files:
                if filename.endswith('.dat'):
                    filepath = os.path.join(root, filename)
                    df = pd.read_csv(filepath, sep=' ')
                    dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def main(dir):
    df = read_data_frame(dir)
    M = max(df['evaluations'])
    with open(os.path.join(dir, 'processed.csv'), 'w') as file:
        print('iteration', 'mean', 'std', file=file)
        for i in range(1, M + 1):
            vals = df.loc[df['evaluations'] == i]['raw_y']
            print(i, np.mean(vals), np.std(vals), file=file)


if __name__ == '__main__':
    main(*sys.argv[1:])
