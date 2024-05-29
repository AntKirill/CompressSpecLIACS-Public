import sys
import os
import pandas as pd
import numpy as np
import utils
import utilsV3


# 
# Create archive of high-performing solutions with pairwise distance greater than TH_DIST
# Consideres all solutions logged to populations.csv files in the given experiment_root
#

def read_data_frame(experiment_root, th_value):
    designs = []
    obj_values = []
    for item in os.listdir(experiment_root):
        fullpath = os.path.join(experiment_root, item)
        if os.path.isdir(fullpath) and item.startswith('everyeval'):
            for item2 in os.listdir(fullpath):
                fullpath2 = os.path.join(fullpath, item2)
                if os.path.isfile(fullpath2) and item2 == 'populations.csv':
                    df = pd.read_csv(fullpath2, sep=' ')
                    good = df.loc[(df['value'] < th_value) & (df['sample_size'] >= 2000)]
                    obj_values = np.concatenate((obj_values, good['value'].tolist()), axis=0)
                    new_designs = good.loc[:, 'x0':'x639']
                    for iter, row in new_designs.iterrows():
                        designs.append(row.tolist())
    return designs, obj_values

def main(dir):
    config = utilsV3.Config()
    config.d0_method = '2'
    config.d1_method = 'kirill'
    config.n_segms = 16
    TH_DIST = 0.05 # d0_method = '2', d1_method = 'kirill'
    TH_OBJ = 0.0003 # sron_guess_obj / 2

    D = utilsV3.CriteriaD()
    F = utilsV3.CriteriaF(D)
    PFR = utilsV3.ReducedDimObjFunSRON(16, F)
    dist_matrix = utils.create_dist_matrix(PFR, config.d0_method)
    dist = utils.create_distance(PFR, dist_matrix, config.d1_method)
    dim_reducer = utils.SegmentsDimReduction(640, config.n_segms)

    print('Locating independent set...', flush=True)
    designs, obj_values = read_data_frame(dir, TH_OBJ)
    sorted_ids = np.argsort(obj_values)
    independent_ids = []

    for i in sorted_ids:
        to_add = True
        for id_ in independent_ids:
            if dist(dim_reducer.to_reduced(designs[i]), dim_reducer.to_reduced(designs[id_])) < TH_DIST:
                to_add = False
                break
        if to_add:
            independent_ids.append(i)
    
    with open(os.path.join(dir, 'archive.csv'), 'w') as file:
        for i in independent_ids:
            print(obj_values[i], *designs[i], file=file)


if __name__ == '__main__':
    main(*sys.argv[1:])
