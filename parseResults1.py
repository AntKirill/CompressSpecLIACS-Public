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
        instance = int(os.path.basename(everyeval).split('-')[1])
        for root, directories, files in os.walk(everyeval):
            for filename in files:
                if filename.endswith('.dat'):
                    filepath = os.path.join(root, filename)
                    df = pd.read_csv(filepath, sep=' ')
                    df['instance'] = instance
                    dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main(dir):
    df = read_data_frame(dir)
    M = max(df['evaluations'])
    NINCTANCES = max(df['instance'].to_numpy(dtype=int)) + 1
    MAXCNT = max(df.groupby('evaluations')['evaluations'].count())
    NINF = float("-inf")
    INF = float("inf")
    best_so_far = np.full((M + 1, NINCTANCES), NINF)
    for i in range(NINCTANCES):
        best_so_far[0][i] = INF
    with open(os.path.join(dir, 'processed.csv'), 'w') as file:
        # argmin_components = [f'argmin_x{i}' for i in range(640)]
        evals_in_instance_headers = [f'eval_in_instance_{i}' for i in range(MAXCNT)]
        print('experiment_root', 'iteration', 'mean', 'std', 'mean_best_so_far', 'std_best_so_far', 'cnt', *evals_in_instance_headers, file=file)
        for iteration in range(1, M + 1):
            lc = df.loc[df['evaluations'] == iteration]
            vals = lc['raw_y'].to_numpy()
            instances = lc['instance'].to_numpy(dtype=int)
            instance_cur_value = np.full(MAXCNT, INF)
            for j in instances:
                instance_cur_value[j] = lc.loc[lc['instance'] == j]['raw_y'].to_numpy(dtype=float)[0]
                best_so_far[iteration][j] = min(best_so_far[iteration - 1][j], instance_cur_value[j])
            iteration_best_so_far = [v for v in best_so_far[iteration] if v != NINF]
            mean_best_so_far = np.mean(iteration_best_so_far)
            std_best_so_far = np.std(iteration_best_so_far)
            # argmin = np.argmin(vals)
            # best = lc.loc[:, 'x0':'x639'].iloc[argmin].to_numpy(dtype=int) #.loc[:, 'x0':'x639']
            print(dir, iteration, np.mean(vals), np.std(vals), mean_best_so_far, std_best_so_far, len(vals), *instance_cur_value, file=file)


if __name__ == '__main__':
    main(*sys.argv[1:])
