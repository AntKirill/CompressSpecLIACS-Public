import os
import numpy as np

import utils
from myrandom import RandomEngine
from utils import Individual


def log_distribution(p, gen_number):
    with open(os.path.join(utils.logger.folder_name, 'umda_distr.txt'), 'a') as f:
        print(f'Generation {gen_number}, sz x crd', file=f)
        for i in range(len(p)):
            print(*p[i], sep=' ', file=f)
        print('', flush=True, file=f)


def umda_Zn_minimization(sz, crd, mu_, lambda_, f, term, is_log_p=False):
    p = np.full((sz, crd), 1. / crd)
    lb = 1 / ((crd - 1) * sz)
    ub = 1. - lb
    iteration = 0
    spent_budget = 0
    best_fitness = float("inf")
    sol = None
    gen_number = 1
    while True:
        if is_log_p:
            log_distribution(p, gen_number)
        if term(iteration, spent_budget, best_fitness):
            break
        pop = []
        for i in range(lambda_):
            x = np.zeros(sz, dtype=int)
            for j in range(sz):
                x[j] = RandomEngine.sample_discrete_dist(p[j])
            obj_value = f(x)
            spent_budget += 1
            pop.append(Individual(x, obj_value))
        pop.sort(key=lambda ind: ind.obj_value)
        if pop[0].obj_value < best_fitness:
            sol = pop[0]
            best_fitness = pop[0].obj_value
        # print(best_fitness)
        for i in range(sz):
            cnt = np.zeros(crd, dtype=int)
            for j in range(mu_):
                cnt[pop[j].genotype[i]] += 1
            for j in range(crd):
                p[i][j] = min(max(cnt[j] / mu_, lb), ub)
        gen_number += 1
    return sol.genotype, sol.obj_value
