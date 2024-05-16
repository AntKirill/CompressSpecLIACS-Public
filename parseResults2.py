import sys
import os
import pandas as pd
import numpy as np

# 
# Parses file population.csv in the given experiment_root
#

def read_data_frame(experiment_root):
    dfs = []
    max_pop_number = float("-inf")
    for item in os.listdir(experiment_root):
        fullpath = os.path.join(experiment_root, item)
        if os.path.isdir(fullpath) and item.startswith('everyeval'):
            for item2 in os.listdir(fullpath):
                fullpath2 = os.path.join(fullpath, item2)
                if os.path.isfile(fullpath2) and item2 == 'populations.csv':
                    df = pd.read_csv(fullpath2, sep=' ')
                    max_pop_number = max(max_pop_number, max(df['pop_number_current']))
                    dfs.append(df)
    return dfs, max_pop_number

def main(dir):
    dfs, M = read_data_frame(dir)    
    with open(os.path.join(dir, 'processed.csv'), 'w') as file:
        print('population', 'mean', 'std', 'cnt', file=file)
        for i in range(0, M + 1):
            values = []
            for df in dfs:
                seq = df.loc[df['pop_number_current'] == i]['value']
                if len(seq) > 0:
                    values.append(min(seq))
            print(i, np.mean(values), np.std(values), len(values), file=file)


if __name__ == '__main__':
    main(*sys.argv[1:])
