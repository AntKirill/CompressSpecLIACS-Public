import sys
import os
import pandas as pd
import numpy as np

# 
# Parses all files with all evals in the given experiment_root
# Extracts information about the whole experiment
#

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
        argmin_components = [f'argmin_x{i}' for i in range(640)]
        print('iteration', 'mean', 'std', 'cnt', 'min', *argmin_components, file=file)
        for i in range(1, M + 1):
            lc = df.loc[df['evaluations'] == i]
            vals = lc['raw_y'].to_numpy()
            argmin = np.argmin(vals)
            best = lc.loc[:, 'x0':'x639'].iloc[argmin].to_numpy(dtype=int) #.loc[:, 'x0':'x639']
            print(i, np.mean(vals), np.std(vals), len(vals), np.min(vals), *best, file=file)


if __name__ == '__main__':
    main(*sys.argv[1:])
