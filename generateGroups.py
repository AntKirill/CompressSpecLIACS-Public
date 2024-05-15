import utilsV3
import utils
import neighbors
from myrandom import RandomEngine
import os
import numpy as np
import sys


T = 32
M = 100
K = T
RDIM = 16


def generate_neighbours(parent, neighbours_method):
    if neighbours_method == 'hamming':
        return neighbors.generate_single_comp_changes(parent, RDIM, K)
    elif neighbours_method == 'dist_ea':
        config = utilsV3.Config()
        config.d0_method = '3'
        config.d1_method = 'kirill'
        D = utilsV3.CriteriaD()
        F = utilsV3.CriteriaF(D)
        dist_matrix = utils.create_dist_matrix(F, config.d0_method)
        dist_matrix_sorted = utilsV3.sort_dist_matrix(dist_matrix)
        dist = utils.create_distance(F, dist_matrix, config.d1_method)
        mutator = utilsV3.DDMutationEA(dist, dist_matrix_sorted)
        mutants = []
        dMin = utilsV3.findExtremeDist(parent, dist_matrix_sorted, dist, 'min', mutator, config.d0_method, config.d1_method)
        new_d_min = 100*dMin
        for i in range(K):
            step = np.random.uniform(10*new_d_min, 15*new_d_min)
            new = mutator.mutation(parent, step).tolist()
            newCopy = new.copy()
            newCopy.sort()
            if not newCopy in mutants:
                print(step, dist(parent, new))
                mutants.append(new)
        return mutants


def main(neighbours_method):
    if not neighbours_method in ['hamming', 'dist_ea']:
        raise ValueError(f'No such method {neighbours_method}')
    experiment_root = './'

    neighbours_root = os.path.join(experiment_root, f'groups_{neighbours_method}_3')
    os.makedirs(neighbours_root, exist_ok=True)
    dim_reducer = utils.SegmentsDimReduction(640, RDIM)
    for i in range(T):
        x = utils.generate_random_solution(RDIM, 4374)
        with open(os.path.join(neighbours_root, f'group_{i}.csv'), 'w') as file:
            print('parent', *dim_reducer.to_original(x), file=file)
            neighs = generate_neighbours(x, neighbours_method)
            print(len(neighs))
            for j, neigh in enumerate(neighs):
                print(f'neighbour_{j}', *dim_reducer.to_original(neigh), file=file)

if __name__ == '__main__':
    main(*sys.argv[1:])
