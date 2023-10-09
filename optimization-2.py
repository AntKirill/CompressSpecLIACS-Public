import argparse
import os

import numpy as np
import random
import utils
from dataclasses import dataclass
from dataclasses_json import dataclass_json


class RandomEngine:
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)

    def sample_uniform01(self):
        return random.random()

    def sample_simmetrical_Bernoulli(self):
        return random.randint(0, 1)

    def sample_Bernoulli(self, success_p):
        return random.random() < success_p

    def sample_int_uniform(self, from_incl, to_excl):
        return random.randint(from_incl, to_excl + 1)

    def sample_Binomial(self, n, success_p):
        success = 0
        for i in range(n):
            if self.sample_uniform01() < success_p:
                success += 1
        return success

    def sample_discrete_dist(self, p):
        r = self.sample_uniform01()
        cdf = 0
        for i in range(1, len(p) + 1):
            cdf = cdf + p[i - 1]
            if cdf > r:
                return i - 1

    class TruncatedExponentialDistribution:
        def __init__(self):
            self.a = 0.
            self.b = 1.
            self.lambda0 = 0.
            self.lambda1 = 0.

        def build(self, m, acc):
            lb = -1. / m
            ub = -float(acc)
            while ub - lb >= acc:
                lambda_ = (lb + ub) / 2.
                left = np.exp(lambda_) * (lambda_ - 1) / (np.exp(lambda_) - 1) + 1 / (np.exp(lambda_) - 1)
                right = m * lambda_
                if left >= right:
                    lb = lambda_
                else:
                    ub = lambda_
            self.lambda1 = lb
            self.lambda0 = np.log(self.lambda1 / (np.exp(self.lambda1) - 1))
            return self

        def pdf(self, x):
            return np.exp(self.lambda0 + self.lambda1 * x)

        def inverse_cdf(self, p):
            return (np.log(self.lambda1 * p + np.exp(self.lambda0)) - self.lambda0) / self.lambda1

        def sample(self):
            p = random.random()
            return self.inverse_cdf(p)


class Individual:
    def __init__(self, genotype, obj_value):
        self.genotype = genotype
        self.obj_value = obj_value


def generate_random_solution(seq_length, lib_size):
    return np.random.randint(0, lib_size, seq_length)


def log_distribution(p, gen_number):
    with open(os.path.join(utils.logger.folder_name, 'umda_distr.txt'), 'a') as f:
        print(f'Generation {gen_number}, sz x crd', file=f)
        for i in range(len(p)):
            print(*p[i], sep=' ', file=f)
        print('', flush=True, file=f)


def umda_Znk_minimization(sz, crd, mu_, lambda_, f, term, is_log_p=False):
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
                x[j] = rnd.sample_discrete_dist(p[j])
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


class FilterSequenceDDMutation:
    def __init__(self, seq_length: int, lib_size: int, dist, config):
        self.seq_length = seq_length
        self.lib_size = lib_size
        self.dist = dist
        self.config = config
        global logger
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
            z, value = umda_Znk_minimization(self.seq_length, self.lib_size, self.config.mu_explore,
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
            z, value = umda_Znk_minimization(self.seq_length, self.lib_size, self.config.mu_explore,
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
        arg, self.step_error = umda_Znk_minimization(self.seq_length, self.lib_size, self.config.mu_mutation,
                                                     self.config.lambda_mutation, f, is_terminate)
        return arg, self.step_error


class ObjFunSRON:
    def __init__(self, rep):
        import instrumentsimulator
        instrument_settings = instrumentsimulator.InstrumentSettings()
        self.instrument = instrumentsimulator.InstrumentSimulator(instrument_settings)
        self.rep = rep
        self.search_space_dim = 640

    def __call__(self, original):
        self.x = np.zeros(self.rep)
        for i in range(self.rep):
            _, self.x[i], = self.instrument.simulateMeasurement(original, nCH4=2000, albedo=0.15, sza=70, n=1,
                                                                verbose=False)
        return np.mean(self.x ** 2)

    @property
    def sron_bias(self):
        return 100 * np.nanmean(self.x)

    @property
    def sron_precision(self):
        return 100 * np.std(self.x)


class ProfiledObjFunSRON:
    def __init__(self, of, config):
        import utils
        self.of = of
        self.obj_f_wrapped = utils.add_logger(self.of, of.search_space_dim, config.folder_name, config.algorithm,
                                              config.algorithm_info)

    def __call__(self, x):
        return self.obj_f_wrapped(x)

    @property
    def instrument(self):
        return self.of.instrument

    @property
    def search_space_dim(self):
        return self.of.search_space_dim


class ReducedDimObjFunSRON:
    def __init__(self, l: int, of):
        import utils
        self.of = of
        self.dim_red = utils.SegmentsDimReduction(of.search_space_dim, l)

    def __call__(self, x):
        y = self.dim_red.to_original(x)
        return self.of(y)

    @property
    def search_space_dim(self):
        return self.dim_red.reduced_dim

    @property
    def instrument(self):
        return self.of.instrument


def create_profiled_obj_fun_for_reduced_space(config):
    return ReducedDimObjFunSRON(config.n_segms, ProfiledObjFunSRON(ObjFunSRON(config.n_reps), config))


def create_dist_matrix(F, d0_method):
    import utils
    return utils.FilterDistanceFactory(F.instrument).create_precomputed_filter_distance_matrix(d0_method,
                                                                                               f'precomputedFiltersDists/method{d0_method}.txt')


def create_distance(F, dist_matrix, d1_method):
    import utils
    d0 = lambda s1, s2: dist_matrix[int(s1), int(s2)]
    return utils.SequenceDistanceFactory(d0=d0, instrument=F.instrument, M=640,
                                         R=F.search_space_dim).create_sequence_distance(d1_method)


class FilterSequenceOptimization:
    def __init__(self, F, dist, config):
        self.F = F
        self.seq_length = F.search_space_dim
        self.lib_size = F.instrument.filterlibrarysize
        self.dist = dist
        self.config = config

    def explore(self):
        self.dd_mutation = FilterSequenceDDMutation(self.seq_length, self.lib_size, self.dist, self.config)
        self.dd_mutation.initialize()

    def __call__(self, initial=None):
        """
        Minimization of objective function
        """
        if initial is None:
            initial = [generate_random_solution(self.seq_length, self.lib_size) for _ in range(self.config.mu_)]
        self.explore()
        self.population = [Individual(ind, self.F(ind)) for ind in initial]
        generations_number = (self.config.budget - self.config.mu_) // self.config.lambda_
        for self.iteration in range(generations_number):
            self.next_population = []
            for _ in range(self.config.lambda_):
                parent1, parent2 = self.choose(self.population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                obj_value = self.F(offspring)
                self.next_population.append(Individual(offspring, obj_value))
            self.population = self.survival_selection(
                self.population, self.next_population, self.config.mu_)
        self.population.sort(key=lambda ind: ind.obj_value)
        return self.population[0]

    def choose(self, population):
        return np.random.choice(population, 2, replace=False)

    def crossover(self, parent1, parent2):
        offspring = np.copy(parent1.genotype)
        for i in range(len(offspring)):
            rnd = np.random.randint(0, 2)
            if rnd == 1:
                offspring[i] = parent2.genotype[i]
        return offspring

    def mutation(self, ind):
        mutant, d = self.dd_mutation.make_mutation(ind, self.config.budget_mutation)
        print('step err', d)
        return mutant

    def survival_selection(self, population, new_population, mu_):
        all_population = np.concatenate((population, new_population)).tolist()
        all_population.sort(key=lambda ind: ind.obj_value)
        return all_population[:mu_]


rnd = RandomEngine()


# %%
def create_bbob_function(f_id, dim, instance, crd):
    import cocoex
    problem = cocoex.Problem().create_problem_bbob_mixint(f_id, dim, instance, np.array([crd]))
    argbest = problem.best_parameter
    valbest = problem(argbest)
    return lambda x: problem(x) - valbest, argbest


def run_umbda(f_id, dim, instance, crd, mu_, lambda_, term):
    f, argbest = create_bbob_function(f_id, dim, instance, crd)
    arg, value = umda_Znk_minimization(dim, crd, mu_, lambda_, f, term)
    return value


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
    budget: int = 5100
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

    @staticmethod
    def implemented_algorithms():
        return frozenset(['ga', 'umda'])

    def validate(self):
        if self.algorithm not in Config.implemented_algorithms():
            raise ValueError(f'Invalid algorithm {self.algorithm}. Valid ones are: {self.__algorithms}')
        return True


def run_optimization(config: Config):
    config.validate()
    F = create_profiled_obj_fun_for_reduced_space(config)
    global logger
    utils.logger.log_config(config)
    if config.algorithm == 'ga':
        dist_matrix = create_dist_matrix(F, config.d0_method)
        dist = create_distance(F, dist_matrix, config.d1_method)
        opt = FilterSequenceOptimization(F, dist, config)
        opt()
    elif config.algorithm == 'umda':
        umda_Znk_minimization(config.n_segms, F.instrument.filterlibrarysize, config.mu_, config.lambda_, F,
                              lambda i1, s, i2: s > config.budget, config.is_log_distr_umda)


# %%


def main():
    parser = argparse.ArgumentParser(description='Runs optimization of the function by SRON with configuration')
    parser.add_argument('-i', '--algorithm_info', help='Information of the optimization algorithm', default='info')
    parser.add_argument('-r', '--n_reps', help='Number of resampling per point', type=int, default=1000)
    parser.add_argument('-m', '--mu', help='Number of parents in EA', type=int, default=10)
    parser.add_argument('-l', '--lambda_', help='Number of offspring in EA', type=int, default=20)
    parser.add_argument('-d0', '--d0_method', help='Distance between filters', default='2')
    parser.add_argument('-d1', '--d1_method', help='Distance between sequences of filters', default='kirill')
    parser.add_argument('--mu_explore', help='Number of parents in UMDA during exploration', type=int, default=10)
    parser.add_argument('--lambda_explore', help='Number of offspring in UMDA during exploration', type=int, default=100)
    parser.add_argument('--budget_explore', help='Max number of distance evals during exploration', type=int, default=2000)
    parser.add_argument('--mu_mutation', help='Number of parents in UMDA during mutation', type=int, default=10)
    parser.add_argument('--lambda_mutation', help='Number of offspring in UMDA during mutation', type=int, default=100)
    parser.add_argument('--budget_mutation', help='Max number of distances evals during mutation', type=int, default=2000)
    parser.add_argument('--seq_length', help='Target length of the sequence of filters', type=int, default=640)
    parser.add_argument('--log_distr', help='Flag to print distribution for every generation in UMDA', type=bool, default=False)
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-a', '--algorithm', help='Optimization algorithm', required=True,
                                choices=Config.implemented_algorithms())
    required_named.add_argument('-f', '--folder_name', help='Name of the folder with logs', required=True)
    required_named.add_argument('-n', '--n_segms', help='Number of segments', type=int, required=True)
    required_named.add_argument('-b', '--budget', help='Max number of obj function evals', type=int, default=5000)
    args = parser.parse_args()

    run_optimization(
        Config(algorithm=args.algorithm,
               algorithm_info=args.algorithm_info,
               folder_name=args.folder_name,
               n_segms=args.n_segms,
               n_reps=args.n_reps,
               mu_=args.mu,
               lambda_=args.lambda_,
               budget=args.budget,
               d0_method=args.d0_method,
               d1_method=args.d1_method,
               mu_explore=args.mu_explore,
               lambda_explore=args.lambda_explore,
               budget_explore=args.budget_explore,
               mu_mutation=args.mu_mutation,
               lambda_mutation=args.lambda_mutation,
               budget_mutation=args.budget_mutation,
               seq_length=args.seq_length,
               is_log_distr_umda=args.log_distr))


if __name__ == '__main__':
    main()
