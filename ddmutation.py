import numpy as np

import utils
from myrandom import RandomEngine
from umda import umda_Zn_minimization
from utils import generate_random_solution


class FilterSequenceDDMutation:
    def __init__(self, seq_length: int, lib_size: int, dist, config):
        self.seq_length = seq_length
        self.lib_size = lib_size
        self.dist = dist
        self.config = config
        if hasattr(utils, 'logger'):
            utils.logger.watch(self, ['dSmallest', 'dLargest', 'm', 'step_size', 'step_error'])

    def findSmallestDist(self, x, budget: int):
        t = 2.
        z = generate_random_solution(self.seq_length, self.lib_size)
        cur_dist = self.dist(x, z)
        spent_budget = 0
        while cur_dist > 0 and spent_budget < budget:
            s = cur_dist / t
            f = lambda y: self.dist(y, x) - s
            term = lambda it_num, spent, objv: spent >= self.config.budget_explore or (objv <= 0 and objv > -s)
            z, value = umda_Zn_minimization(self.seq_length, self.lib_size, self.config.mu_explore,
                                            self.config.lambda_explore, f, term)
            cur_dist = min(self.dist(x, z), cur_dist)
            print(cur_dist)
            spent_budget += 1
        return cur_dist

    def findLargestDist(self, x, budget: int):
        t = 1.2
        z = generate_random_solution(self.seq_length, self.lib_size)
        cur_dist = self.dist(x, z)
        spent_budget = 0
        while cur_dist > 0 and spent_budget < budget:
            s = cur_dist * t
            f = lambda y: s - self.dist(y, x)
            term = lambda it_num, spent, objv: spent >= self.config.budget_explore or objv <= 0
            z, value = umda_Zn_minimization(self.seq_length, self.lib_size, self.config.mu_explore,
                                            self.config.lambda_explore, f, term)
            cur_dist = max(self.dist(x, z), cur_dist)
            print(cur_dist)
            spent_budget += 1
        return cur_dist

    def findDistGamma(self, dMin, dMax):
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
        return (lbGamma + ubGamma) / 2.

    def createDistribution(self):
        return RandomEngine.TruncatedExponentialDistribution().build(self.m, 1e-9)

    def initialize(self, known=None):
        if known:
            self.dSmallest, self.dLargest, self.gamma = known
        else:
            x = generate_random_solution(self.seq_length, self.lib_size)
            self.dSmallest = self.findSmallestDist(x, 10)
            self.dLargest = self.findLargestDist(x, 10)
            self.gamma = self.findDistGamma(self.dSmallest, self.dLargest)
        self.m = 2 * self.dSmallest
        self.D = self.createDistribution()

    def make_mutation(self, x, budget: int):
        s = self.D.sample()
        self.step_size = np.sqrt(-np.log(1 - s)) / self.gamma
        f = lambda y: abs(self.dist(x, y) - self.step_size)
        is_terminate = lambda it_number, spent_budget, obj_value: spent_budget >= budget or obj_value == 0
        return self.internal_optimization(f, is_terminate)

    def internal_optimization(self, f, is_terminate):
        arg, self.step_error = umda_Zn_minimization(self.seq_length, self.lib_size, self.config.mu_mutation,
                                                    self.config.lambda_mutation, f, is_terminate)
        return arg, self.step_error
