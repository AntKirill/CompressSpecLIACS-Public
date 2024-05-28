import sys
import instrumentsimulator
import numpy as np
from importlib import reload
import objf
from scipy import stats
import utils as utils
import mylogger as mylogger
import algorithms as algs
from myrandom import RandomEngine
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from abc import ABC, abstractmethod
import algorithms as algs
import umda
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from collections import namedtuple


class CriteriaD:
    def __init__(self, extended=True):
        instrument_settings = instrumentsimulator.InstrumentSettings()
        self.instrument = instrumentsimulator.InstrumentSimulator(instrument_settings)
        self.search_space_dim = 640
        self.x = None
        self.extended = extended
        self.called_count = 0

    def __call__(self, original):
        self.called_count += 1
        _, value, = self.instrument.simulateMeasurement(original, nCH4=2000, albedo=0.15, sza=70, n=1, extended=self.extended, verbose=False)
        return value
    
    def get_called_count(self):
        return self.called_count


class CriteriaF:
    def __init__(self, D):
        self.D = D
        self.instrument = D.instrument
        self.values = None
        self.search_space_dim = 640

    def __call__(self, x, reps):
        self.values = np.zeros(reps)
        for i in range(reps):
            self.values[i] = self.D(x)
        return np.mean(self.values**2)

    def get_measurements(self):
        return self.values
    
    def get_called_count(self):
        return self.D.get_called_count()


class ProfiledF:
    def __init__(self, of, config):
        reload(mylogger)
        reload(utils)
        self.of = of
        self.obj_f_wrapped = utils.add_logger(self.of, of.search_space_dim, config.folder_name, config.algorithm, config.algorithm_info, config.instance)
        utils.logger.watch(self, ['DxMeanSqr', 'DxVar'])

    def __call__(self, x, reps):
        return self.obj_f_wrapped(x, reps)

    def get_measurements(self):
        return self.of.get_measurements()

    @property
    def instrument(self):
        return self.of.instrument

    @property
    def search_space_dim(self):
        return self.of.search_space_dim

    @property
    def DxMeanSqr(self):
        return np.mean(self.get_measurements())**2

    @property
    def DxVar(self):
        return np.var(self.get_measurements())
    
    def get_called_count(self):
        return self.of.get_called_count()


class ReducedDimObjFunSRON:
    def __init__(self, l: int, of):
        import utils
        self.of = of
        self.dim_red = utils.SegmentsDimReduction(of.search_space_dim, l)

    def __call__(self, x, reps):
        y = self.dim_red.to_original(x)
        return self.of(y, reps)

    def get_measurements(self):
        return self.of.get_measurements()

    @property
    def search_space_dim(self):
        return self.dim_red.reduced_dim

    @property
    def instrument(self):
        return self.of.instrument

    def get_called_count(self):
        return self.of.get_called_count()


def sort_dist_matrix(matrix):
    matrix_sorted_rows = []
    for i in range(len(matrix)):
        tmp = [(matrix[i][j], j) for j in range(len(matrix[i]))]
        tmp.sort()
        matrix_sorted_rows.append(tmp)
    return matrix_sorted_rows


def on_reload():
    global D, PFR, L, Fr, x0, dist_matrix, dist_matrix_sorted, dist
    D = CriteriaD()
    PFR = CriteriaF(D)
    L = D.instrument.filterlibrarysize
    Fr = ReducedDimObjFunSRON(16, PFR)
    x0 = D.instrument.filterguess()
    dist_matrix = utils.create_dist_matrix(PFR, 2)
    dist_matrix_sorted = sort_dist_matrix(dist_matrix)
    dist = utils.create_distance(PFR, dist_matrix, 'kirill')


def umda_Zn_minimization(sz, crd, mu_, lambda_, f, term):
    p = np.full((sz, crd), 1. / crd)
    lb = 1 / ((crd - 1) * sz)
    ub = 1. - lb
    iteration = 0
    spent_budget = 0
    best_fitness = float("inf")
    sol = None
    gen_number = 1
    NoiseFreeInd = namedtuple('NoiseFreeInd', ['genotype', 'obj_value'])
    while True:
        if term(iteration, spent_budget, best_fitness):
            break
        pop = []
        for i in range(lambda_):
            x = np.zeros(sz, dtype=int)
            for j in range(sz):
                x[j] = RandomEngine.sample_discrete_dist(p[j])
            obj_value = f(x)
            spent_budget += 1
            pop.append(NoiseFreeInd(x, obj_value))
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


class DDMutation(ABC):
    @abstractmethod
    def mutation(self, x, s):
        pass


class DDMutationEA(DDMutation):
    def __init__(self, dist, dist_matrix_sorted):
        self.budget = 1000
        self.lambda_ = 5
        self.dist = dist
        self.dist_matrix_sorted = dist_matrix_sorted

    def mutation(self, x, s):
        ea = algs.PermutationSeqDistanceInverse(self.budget, 1, self.lambda_, x, self.dist_matrix_sorted, s)
        return ea(self.dist)
    

class DDMutationUMDA(DDMutation):
    def __init__(self, dist, L):
        self.my_config = Config()
        self.my_config.mu_ = 50
        self.my_config.lambda_ = 200
        self.my_config.budget = 1000
        self.dist = dist
        self.L = L

    def mutation(self, x, s):
        y, value = umda_Zn_minimization(len(x), self.L, self.my_config.mu_, self.my_config.lambda_, lambda y: abs(self.dist(y, x) - s), lambda i1, bdg, i2: bdg >= self.my_config.budget)
        print('Mutation step ordered', s, 'Mutation step made', value)
        # F = lambda y, n: abs(self.dist(y, x) - s)
        # opt = optimizationV3.UMDA(F, None, None, self.my_config, self.my_config.n_segms, self.L)
        # y, value = opt()
        # print(s, value)
        return y
    

def findExtremeDist(x, matrix_sorted_rows, dist, extreme_str, mutator, d0_type, d1_type):
    if d0_type == '2' and d1_type == 'kirill' and len(x) == 16 and extreme_str == 'min':
        return 1.3390909280042163e-07
    if d0_type == '2' and d1_type == 'kirill' and len(x) == 16 and extreme_str == 'max':
        return 0.0922
    if d0_type == '3' and d1_type == 'kirill' and len(x) == 16 and extreme_str == 'min':
        return 4.017905652392746e-08
    if d0_type == '3' and d1_type == 'kirill' and len(x) == 16 and extreme_str == 'max':
        return 6.107193554680426
    if extreme_str == 'min':
        pos, min_dist, arg_min_dist = None, float("inf"), None
        for i, xi in enumerate(x):
            cur_dist, arg_dist = matrix_sorted_rows[xi][1]
            if cur_dist < min_dist:
                pos, min_dist, arg_min_dist = i, cur_dist, arg_dist
        x1 = np.copy(x)
        x1[pos] = arg_min_dist
        return dist(x, x1)
    elif extreme_str == 'max':
        x1 = utils.generate_random_solution(len(x), 4374)
        d = dist(x, x1)
        cnt = 0
        max_d, arg_max_d = d, x1
        while cnt < 2:
            target_d = d * 1.2
            y = mutator.mutation(x, target_d)
            dxy = dist(x, y)
            if dxy >= target_d:
                d = dxy
                cnt = 0
            else:
                cnt += 1
            if dxy > max_d:
                max_d, arg_max_d = dxy, y
            print(max_d)
        return dist(x, arg_max_d)
    else:
        raise ValueError(f'arg extreme is either min or max, but {extreme_str} is passed')


def findDistGamma(dMin, dMax):
    print(dMin, dMax)
    delta = 1e-4
    dd = (dMax / dMin) ** 2.
    eps1 = delta
    eps2 = 0.
    dEpsMin = float("inf")
    while eps1 <= 0.01:
        eps2LB = (1. - eps1) ** dd
        dEps = eps2LB - eps1
        if dEps < dEpsMin:
            dEpsMin = dEps
            eps2 = (eps1 + eps2LB) / 2.
        if dEpsMin <= 0.:
            break
        eps1 += delta
    lbGamma = np.sqrt(-np.log(eps2)) / dMax
    ubGamma = np.sqrt(-np.log(1. - eps1)) / dMin
    print('eps1', eps1, 'eps2', eps2)
    return (lbGamma + ubGamma) / 2.


def createDistribution(m):
    return RandomEngine.TruncatedExponentialDistribution().build(m, 1e-9)


@dataclass_json
@dataclass
class Config:
    algorithm: str = 'dd-ga'
    algorithm_info: str = 'info'
    folder_name: str = 'exp'
    n_segms: int = 16
    n_reps: int = 1000
    mu_: int = 10
    lambda_: int = 20
    budget: int = 2100
    d0_method: str = '2'
    d1_method: str = 'kirill'
    mu_explore: int = 10
    lambda_explore: int = 100
    budget_explore: int = 2000
    mu_mutation: int = 10
    lambda_mutation: int = 100
    budget_mutation: int = 2000
    seq_length: int = 640
    is_log_distr_umda: bool = False
    dd_mutation: str = 'ea'
    instance: int = 0
    robustness: str = 'fixed'
    max_samples: int = 5000
    significance: float = 0.05
    sample_inc: int = 100
    mut_rate: float = 1
    min_distance_scaling: float = 100
    is_extended: bool = True # In the earlier experiments we used False

    @staticmethod
    def implemented_algorithms():
        return frozenset(['dd-ga', 
                          'dd-opll', 
                          'ea-simple', 
                          'ea-simple-cross', 
                          'dd-ls', 
                          'mies', 
                          'ngopt', 
                          'fastga-ng', 
                          'portfolio-ng', 
                          'bo-ng', 
                          'bo-sk',
                          'dd-es', 
                          'umda', 
                          'umda1', 
                          'umda2',
                          'umda2-dist'
                          ])

    @staticmethod
    def supported_dd_mutations():
        return frozenset(['ea', 'umda'])
    
    @staticmethod
    def supported_robustness():
        return frozenset(['fixed', 'welch'])
    
    def add_cml_args(self, parser):
        parser.add_argument('-i', '--algorithm_info', help='Information of the optimization algorithm', default=self.algorithm_info)
        parser.add_argument('--n_reps', help='Number of resampling per point', type=int, default=self.n_reps)
        parser.add_argument('--mu_', help='Number of parents in EA', type=int, default=self.mu_)
        parser.add_argument('--lambda_', help='Number of offspring in EA', type=int, default=self.lambda_)
        parser.add_argument('-d0', '--d0_method', help='Distance between filters', default=self.d0_method)
        parser.add_argument('-d1', '--d1_method', help='Distance between sequences of filters', default=self.d1_method)
        parser.add_argument('--mu_explore', help='Number of parents in UMDA during exploration', type=int, default=self.mu_explore)
        parser.add_argument('--lambda_explore', help='Number of offspring in UMDA during exploration', type=int, default=self.lambda_explore)
        parser.add_argument('--budget_explore', help='Max number of distance evals during exploration', type=int, default=self.budget_explore)
        parser.add_argument('--mu_mutation', help='Number of parents in UMDA during mutation', type=int, default=self.mu_mutation)
        parser.add_argument('--lambda_mutation', help='Number of offspring in UMDA during mutation', type=int, default=self.lambda_mutation)
        parser.add_argument('--budget_mutation', help='Max number of distances evals during mutation', type=int, default=self.budget_mutation)
        parser.add_argument('--seq_length', help='Target length of the sequence of filters', type=int, default=self.seq_length)
        parser.add_argument('--is_log_distr_umda', help='Flag to print distribution for every generation in UMDA', type=bool, default=self.is_log_distr_umda)
        parser.add_argument('--dd_mutation', help='Distance-Driven mutation operator internal optimization', type=str, default=self.dd_mutation, 
                            choices=Config.supported_dd_mutations())
        parser.add_argument('--budget', help='Max number of obj function evals', type=int, default=self.budget)
        parser.add_argument('--robustness', help='Method to ensure statisticall significance when comparing noisy solutions', type=str, default=self.robustness, 
                            choices=Config.supported_robustness())
        parser.add_argument('--max_samples', help='Max number of samples per solution', type=int, default=self.max_samples)
        parser.add_argument('--significance', help='Maximum pvalue in welch ttest to ensure sufficient statistical evidence in ranking', type=float, default=self.significance)
        parser.add_argument('--sample_inc', help='Incrase in the number of samples', type=int, default=self.sample_inc)
        parser.add_argument('--mut_rate', help='Mutation rate for simple EAs', type=float, default=self.mut_rate)
        parser.add_argument('--min_distance_scaling', help='Scales the minimal found distance in the given search space', type=float, default=self.min_distance_scaling)
        parser.add_argument('--is_extended', help='Extended model for calculation objective function', type=bool, default=self.is_extended)
        required_named = parser.add_argument_group('required named arguments')
        required_named.add_argument('-a', '--algorithm', help='Optimization algorithm', required=True, choices=Config.implemented_algorithms())
        required_named.add_argument('--folder_name', help='Name of the folder with logs', required=True)
        required_named.add_argument('--n_segms', help='Number of segments', type=int, required=True)
        required_named.add_argument('--instance', help='Instance number', type=int, required=True)


def create_profiled_obj_fun_for_reduced_space(config):
    D = CriteriaD(config.is_extended)
    F = CriteriaF(D)
    PF = ProfiledF(F, config)
    PFR = ReducedDimObjFunSRON(config.n_segms, PF)
    return PFR
