import sys
import os
import pandas as pd
import numpy as np
import utils
import utilsV3


# 
# Create archive of high-performing solutions with pairwise distance greater than TH_DIST
# Consideres all solutions logged to everyeval-${instance} directories in the given experiment_root
#


def read_data_frame(experiment_root, th_value):
    designs = []
    obj_values = []
    for item in os.listdir(experiment_root):
        fullpath = os.path.join(experiment_root, item)
        if os.path.isdir(fullpath) and item.startswith('everyeval'):
            instance = int(os.path.basename(item).split('-')[1])
            for root, directories, files in os.walk(fullpath):
                for filename in files:
                    if filename.endswith('.dat'):
                        filepath = os.path.join(root, filename)
                        df = pd.read_csv(filepath, sep=' ')
                        good = df.loc[(df['raw_y'] < th_value)]
                        new_obj_values = good['raw_y'].tolist()
                        obj_values += new_obj_values
                        new_designs = good.loc[:, 'x0':'x639']
                        for iter, row in new_designs.iterrows():
                            designs.append(row.tolist())
    return designs, obj_values


def locate_independent_set(dir, obj_values, designs, config, dist, TH_DIST):
    print(len(obj_values))

    dim_reducer = utils.SegmentsDimReduction(640, config.n_segms)

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


def main(dir):
    config = utilsV3.Config()
    config.d0_method = '2'
    config.d1_method = 'kirill'
    config.n_segms = 16
    TH_DIST = 0.05 # d0_method = '2', d1_method = 'kirill'
    TH_OBJ = 0.0003 # sron_guess_obj / 2

    designs, obj_values = read_data_frame(dir, TH_OBJ)
    D = utilsV3.CriteriaD()
    F = utilsV3.CriteriaF(D)
    PFR = utilsV3.ReducedDimObjFunSRON(16, F)
    dist_matrix = utils.create_dist_matrix(PFR, config.d0_method)
    dist = utils.create_distance(PFR, dist_matrix, config.d1_method)
    print('Locating independent set...', flush=True)
    locate_independent_set(dir, obj_values, designs, config, dist, TH_DIST)


if __name__ == '__main__':
    main(*sys.argv[1:])
