import sys
import instrumentsimulator
import numpy as np
from importlib import reload
import objf
from scipy import stats
import utils as utils
import mylogger as mylogger
import algorithms as algs
import umda
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


class CriteriaD:
    def __init__(self):
        instrument_settings = instrumentsimulator.InstrumentSettings()
        self.instrument = instrumentsimulator.InstrumentSimulator(instrument_settings)
        self.search_space_dim = 640
        self.x = None

    def __call__(self, original):
        _, value, = self.instrument.simulateMeasurement(original, nCH4=2000, albedo=0.15, sza=70, n=1, verbose=False)
        return value


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


class ProfiledF:
    def __init__(self, of, config):
        reload(mylogger)
        reload(utils)
        self.of = of
        self.obj_f_wrapped = utils.add_logger(self.of, of.search_space_dim, config.folder_name, config.algorithm, config.algorithm_info)
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


class DDMutation(ABC):
    @abstractmethod
    def mutation(self, x):
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
    def __init__(self, dist):
        self.budget = 1000
        self.dist = dist

    def mutaiton(self, x, s):
        y, value = umda.umda_Zn_minimization(len(x), L, 50, 200, lambda y: abs(self.dist(y, x) - s), lambda i1, bdg, i2: bdg >= self.budget)
        return y
    

def findExtremeDist(x, matrix_sorted_rows, dist, extreme_str, mutator):
    if type(dist) is utils.SequenceDistanceKirill and len(x) == 16 and extreme_str == 'max':
        return 0.0922
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
        x1 = utils.generate_random_solution(len(x), L)
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
    algorithm: str = 'ga'
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
    T: int = 10

    @staticmethod
    def implemented_algorithms():
        return frozenset(['ga', 'umda', 'ea-simple'])

    def supported_dd_mutations():
        return frozenset(['ea', 'umda'])

    def validate(self):
        if self.algorithm not in Config.implemented_algorithms():
            raise ValueError(f'Invalid algorithm {self.algorithm}. Valid ones are: {self.__algorithms}')
        return True


def create_profiled_obj_fun_for_reduced_space(config):
    D = CriteriaD()
    F = CriteriaF(D)
    PF = ProfiledF(F, config)
    PFR = ReducedDimObjFunSRON(16, PF)
    return PFR
