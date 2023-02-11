import math
from abc import ABC, abstractmethod

import cma
import numpy as np

import mipego
import scipy

from utils import FilterDistanceFactory, SequenceDistanceFactory, SequenceDistanceKirill


class Individual:
    def __init__(self, genotype, obj_value):
        self.genotype = genotype
        self.obj_value = obj_value


class AbstractSimulatedAnnealing(ABC):
    def __init__(self, budget, t0=10, a=0.9999):
        self.budget = budget
        self.t0 = t0
        self.a = a
        self.t = None
        self.update_prob = None
        self.cur_obj_value = None

    def __call__(self, f, s_init):
        self.update_prob = None
        self.cur_obj_value = None
        self.t = self.t0
        ind = Individual(s_init, f(s_init))
        best = ind
        for iteration in range(self.budget):
            self.cur_obj_value = ind.obj_value
            self.t = self._update_temperature(self.t)
            x = self._select_neighbour(ind.genotype)
            candidate = Individual(x, f(x))
            if candidate.obj_value < ind.obj_value:
                best = candidate
            if self._get_step_probability(ind.obj_value, candidate.obj_value, self.t) >= np.random.uniform(0, 1):
                ind = candidate
        return best.genotype

    def _update_temperature(self, t):
        return t * self.a

    def _get_step_probability(self, f_old, f_new, t):
        if f_new < f_old:
            self.update_prob = 1.
        else:
            self.update_prob = np.exp((f_old - f_new) * 1000 / t)
        return self.update_prob

    @abstractmethod
    def _select_neighbour(self, x):
        pass

    @property
    def temperature(self):
        return self.t

    @property
    def current_solution_quality(self):
        return self.cur_obj_value

    @property
    def last_update_prob(self):
        return self.update_prob


class FiltersPhenoSimulatedAnnealing(AbstractSimulatedAnnealing):
    def __init__(self, budget, t0, a, generator, neighbourhood_dist):
        super().__init__(budget, t0, a)
        self.generator = generator
        self.distribution = Uniform()
        self.d = neighbourhood_dist

    def _select_neighbour(self, x):
        dist = self.distribution.sample(0., self.d)
        return self.generator.generate_distant_offspring(x, dist)


class RLSSubspaces:
    def __init__(self, mutation, budget, M, L, seed):
        self.mutation = mutation
        self.M = M
        self.L = L
        self.budget_per_subspace = budget
        np.random.seed(seed)

    def get_subspaces(self, x):
        value_to_pos = {}
        for i in range(len(x)):
            if x[i] in value_to_pos:
                value_to_pos[x[i]].append(i)
            else:
                value_to_pos[x[i]] = [i]
        subspaces = []
        for k, v in value_to_pos.items():
            subspaces.append(v)
        return subspaces

    def __call__(self, f, initial_guess):
        subspaces = self.get_subspaces(initial_guess)
        x = initial_guess
        fx = f(x)
        for subspace in subspaces:
            for iteration in range(self.budget_per_subspace):
                i = np.random.randint(len(subspace))
                pos = subspace[i]
                y = x.copy()
                y[pos] = self.mutation(y[pos])
                fy = f(y)
                if fy < fx:
                    x = y
                    fx = fy
        return x


class EA:
    def __init__(self, mutation, budget, M, L, seed):
        self.mutation = mutation
        self.M = M
        self.L = L
        self.budget = budget
        np.random.seed(seed)

    def __call__(self, f, initial_guess):
        x = initial_guess
        fx = f(x)
        n = len(initial_guess)
        for iteration in range(self.budget):
            y = x.copy()
            for i in range(n):
                if np.random.uniform() < 1. / n:
                    y[i] = self.mutation(y[i])
            fy = f(y)
            if fy < fx:
                x = y
                fx = fy
        return x


class NoiseBOWrapper:
    def __init__(self, doe_size, budget, M, L, seed):
        self.M = M
        self.L = L
        self.budget = budget
        self.doe_size = doe_size
        np.random.seed(seed)

    def __call__(self, f):
        I = mipego.OrdinalSpace([0, self.L - 1]) * self.M
        model = mipego.RandomForest(levels=I.levels)
        opt = mipego.NoisyBO(
            search_space=I,
            obj_fun=f,
            model=model,
            max_FEs=self.budget,
            DoE_size=self.doe_size,  # the initial DoE size
            # eval_type='dict',
            acquisition_fun='MGFI',
            acquisition_par={'t': 2},
            n_job=2,  # number of processes
            n_point=2,  # number of the candidate solution proposed in each iteration
            verbose=True  # turn this off, if you prefer no output
        )
        xopt, fopt, stop_dict = opt.run()
        return xopt, fopt


class MIESWrapper:
    def __init__(self, budget, M, L, seed, mu_=4, lambda_=10):
        self.M = M
        self.L = L
        self.budget = budget
        self.mu_ = mu_
        self.lambda_ = lambda_
        np.random.seed(seed)

    def __call__(self, f):
        I = mipego.OrdinalSpace([0, self.L - 1]) * self.M
        mies = mipego.optimizer.mies.MIES(
            search_space=I,
            obj_func=f,
            max_eval=self.budget,
            mu_=self.mu_,
            lambda_=self.lambda_)
        xopt, fopt, stop_dict = mies.optimize()
        return xopt, fopt


def uniform_mutation(k, r):
    num = np.random.randint(r - 1)
    if num == k:
        return r - 1
    return num


class AbstractES(ABC):
    def __init__(self, genotype_space, dim, budget, mu_, lambda_):
        self.genotype_space = genotype_space
        self.dim = dim
        self.budget = budget
        self.mu_ = mu_
        self.lambda_ = lambda_

    @abstractmethod
    def choose(self, population):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutation(self, ind):
        pass

    @abstractmethod
    def survival_selection(self, population, new_population, mu_):
        pass

    def update_parameters(self):
        pass

    def __call__(self, obj_function, initial=None):
        """
        Minimization of objective function
        """
        if initial is None:
            initial = []
            for i in range(self.mu_):
                initial.append(self.genotype_space.sample_uniform())
        self.population = [Individual(ind, obj_function(ind)) for ind in initial]
        generations_number = (self.budget - self.mu_) // self.lambda_
        for self.iteration in range(generations_number):
            self.next_population = []
            for _ in range(self.lambda_):
                parent1, parent2 = self.choose(self.population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                obj_value = obj_function(offspring)
                self.next_population.append(Individual(offspring, obj_value))
            self.population = self.survival_selection(
                self.population, self.next_population, self.mu_)
            self.update_parameters()
        self.population.sort(key=lambda ind: ind.obj_value)
        return self.population[0]


class MyES(AbstractES):

    def __init__(self, genotype_space, dim, budget, mu_, lambda_, initial_distance, generator):
        super().__init__(genotype_space, dim, budget, mu_, lambda_)
        self.d = initial_distance
        self.generator = generator

    def choose(self, population):
        """
        Randomly choose two parents from the current population
        """
        return np.random.choice(population, 2, replace=False)

    def crossover(self, parent1, parent2):
        """
        For every component of the resulting offspring equiprobable pick the component of the first or second parent
        """
        offspring = np.copy(parent1.genotype)
        for i in range(len(offspring)):
            rnd = np.random.randint(0, 2)
            if rnd == 1:
                offspring[i] = parent2.genotype[i]
        return offspring

    def mutation(self, ind):
        """
        Makes step approximately to the given distance
        """
        return self.generator.generate_distant_offspring(ind, self.d)

    def survival_selection(self, population, new_population, mu_):
        """
        Elitist
        """
        all_population = np.concatenate((population, new_population)).tolist()
        all_population.sort(key=lambda ind: ind.obj_value)
        return all_population[:mu_]


class MyDistribution(ABC):
    @abstractmethod
    def sample(self, min_value, max_value):
        pass


class Uniform(MyDistribution):
    def sample(self, min_value, max_value):
        return np.random.uniform(min_value, max_value)


class Beta(MyDistribution):
    def __init__(self) -> None:
        super().__init__()
        self.distribution = scipy.stats.beta(2, 7)

    def sample(self, min_value, max_value):
        sample = min_value + self.distribution.rvs(size=1) * (max_value - min_value)
        return sample[0]


class Normal(MyDistribution):
    def sample(self, min_value, max_value):
        mu = (min_value + max_value) / 2.
        sigma = mu / 10
        return np.random.normal(mu, sigma)


class Exponential(MyDistribution):
    def sample(self, min_value, max_value):
        l = 10.
        x = -np.log(1 - (1 - np.exp(-l)) * np.random.uniform(0, 1.)) / l
        sample = min_value + x * (max_value - min_value)
        return sample


class MyESFixedDistDistribution(MyES):
    def __init__(self, genotype_space, dim, budget, mu_, lambda_, initial_distance, generator, distribution):
        super().__init__(genotype_space, dim, budget, mu_, lambda_, initial_distance, generator)
        self.distribution = distribution
        self.min_dist = 0.
        self.max_dist = 0.1

    def mutation(self, ind):
        self.max_dist = np.random.uniform(0, 0.01)
        self.d = self.distribution.sample(self.min_dist, self.max_dist)
        return super().mutation(ind)


class MyES1(MyES):
    def crossover(self, parent1, parent2):
        offspring = np.copy(parent1.genotype)
        return offspring


def harmonic_sample(n):
    if n <= 0:
        return 0
    c = 0
    for i in range(1, n + 1):
        c += i ** -1
    c = c ** -1
    p = [c / i for i in range(1, n + 1)]
    return np.random.choice([i for i in range(0, n)], p=p)


class PermutationSeqDistanceInverse:
    def __init__(self, budget, mu_, lambda_, initial_individual, matrix_sorted_rows, target_distance):
        self.lambda_ = lambda_
        self.mu_ = mu_
        self.budget = budget
        self.initial_individual = initial_individual
        self.dim = len(initial_individual)
        self.matrix_sorted_rows = matrix_sorted_rows
        self.target_distance = target_distance
        self.L = len(matrix_sorted_rows[0])

    def __obj_function(self, d1, x):
        pi, value = d1.get_permutation_and_value(self.initial_individual, x)
        obj_distance = abs(value - self.target_distance)
        return pi, value, obj_distance

    def mutation(self, parent, parent_obj):
        pi, value, _ = parent_obj
        rep_cnt, max_rep_cnt = 0, 10
        while rep_cnt < max_rep_cnt:
            index_initial = np.random.randint(0, self.dim)
            index_parent = pi[index_initial]
            filter_initial = self.initial_individual[index_initial]
            filter_parent = parent[index_parent]
            pos = None
            for i in range(self.L):
                dist_initial, filter_id = self.matrix_sorted_rows[filter_initial][i]
                if filter_id == filter_parent:
                    pos = i
                    break
            assert pos is not None
            to_pos = pos
            if value < self.target_distance and pos != self.L - 1:
                to_pos = pos + 1 + harmonic_sample(self.L - pos - 1)
            elif value > self.target_distance and pos != 0:
                to_pos = pos - 1 - harmonic_sample(pos)
            to_filter = self.matrix_sorted_rows[filter_initial][to_pos][1]
            if to_filter != filter_parent:
                offspring = np.copy(parent)
                offspring[index_parent] = to_filter
                return offspring
            rep_cnt += 1
        return np.copy(parent)

    def __call__(self, d1: SequenceDistanceKirill):
        parent = np.copy(self.initial_individual)
        parent_obj = ([i for i in range(0, self.dim)], 0., self.target_distance)
        for iteration in range(self.budget // self.lambda_):
            pop = []
            for i in range(self.lambda_):
                offspring = self.mutation(parent, parent_obj)
                offspring_obj = self.__obj_function(d1, offspring)
                pop.append((offspring, offspring_obj))
            candidate = min(pop, key=lambda x: x[1][2])
            if candidate[1][2] < parent_obj[2]:
                parent, parent_obj = candidate
        return parent


class AbstractGenerator(ABC):
    def __init__(self, d1) -> None:
        self.d1 = d1
        self.dist_from_x = None

    @abstractmethod
    def generate_distant_offspring(self, x, d):
        pass

    @property
    def distance_from_parent(self):
        return self.dist_from_x

    @property
    def target_distance_from_parent(self):
        return self.dist_from_x

    @staticmethod
    def _get_matrix_sorted_rows(matrix):
        matrix_sorted_rows = []
        for i in range(len(matrix)):
            tmp = [(matrix[i][j], j) for j in range(len(matrix[i]))]
            tmp.sort()
            matrix_sorted_rows.append(tmp)
        return matrix_sorted_rows


class GeneratorEA(AbstractGenerator):
    def __init__(self, filter_dist_matrix, d1, budget):
        super().__init__(d1)
        self.matrix = filter_dist_matrix
        self.matrix_sorted_rows = self._get_matrix_sorted_rows(self.matrix)
        self.budget = budget
        self.target_dist_from_x = None
        self.distribution = scipy.stats.beta(2, 8)

    def generate_distant_offspring(self, x, d):
        self.target_dist_from_x = d
        ea = PermutationSeqDistanceInverse(self.budget, 1, 5, x, self.matrix_sorted_rows, d)
        y = ea(self.d1)
        self.dist_from_x = self.d1(x, y)
        return y

    @property
    def target_distance_from_parent(self):
        return self.target_dist_from_x


class Generator1(AbstractGenerator):
    def __init__(self, filter_dist_matrix, d1):
        super().__init__(d1)
        self.matrix = filter_dist_matrix
        self.matrix_sorted_rows = self._get_matrix_sorted_rows(self.matrix)

    def generate_distant_offspring(self, x, d):
        max_v = float('-inf')
        best_y = None
        for it in range(10):
            y = np.copy(x)
            for i in range(len(y)):
                ind = int(len(self.matrix_sorted_rows[0]) * np.random.uniform(0, d))
                _, y[i] = self.matrix_sorted_rows[x[i]][ind]
            v = self.d1(x, y)
            if v > max_v:
                max_v, best_y = v, y
        self.dist_from_x = max_v
        return best_y


class Generator2(Generator1):
    def __init__(self, filter_dist_matrix, d1):
        super().__init__(filter_dist_matrix, d1)

    def generate_distant_offspring(self, x, d):
        was = False
        y = np.copy(x)
        for i in range(len(x)):
            if np.random.uniform(0, 1) < d:
                ind = int(len(self.matrix_sorted_rows[0]) * np.random.uniform(0, d))
                _, y[i] = self.matrix_sorted_rows[x[i]][ind]
                was = True
        if not was:
            i = np.random.randint(0, len(x))
            ind = int(len(self.matrix_sorted_rows[0]) * np.random.uniform(0, d))
            _, y[i] = self.matrix_sorted_rows[x[i]][ind]
        self.dist_from_x = self.d1(y, x)
        return y


class GeneratorHarmonic(Generator1):
    def __init__(self, filter_dist_matrix, d1, seqs, offset):
        super().__init__(filter_dist_matrix, d1)
        self.seqs = seqs
        self.dists = np.zeros(len(self.seqs))
        self.offset = offset
        self.sorted_ids = None
        self.pos = None

    def _to_offspring(self, x, seq):
        offspring = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            _, offspring[i] = self.matrix_sorted_rows[x[i]][self.offset + seq[i]]
        return offspring

    def generate_distant_offspring(self, x, d):
        for i in range(len(self.seqs)):
            self.dists[i] = self.d1(x, self._to_offspring(x, self.seqs[i]))
        self.sorted_ids = np.argsort(self.dists)
        r = len(self.dists) - 1
        c = 0
        for i in range(1, r + 1):
            c += i ** -1
        c = c ** -1
        p = [c / i for i in range(1, r + 1)]
        self.pos = np.random.choice([i for i in range(1, r + 1)], p=p)
        offspring = self._to_offspring(x, self.seqs[self.sorted_ids[self.pos]])
        self.dist_from_x = self.dists[self.sorted_ids[self.pos]]
        return offspring


class CombinationsWithRepetitions:
    def __init__(self):
        self.skipped = 0
        self.is_generated = False
        self.seqs = None
        self.a = None

    def __gen(self, pos, el, n, k):
        if pos == k:
            self.seqs.append(np.copy(self.a))
            return
        for i in range(el, n):
            self.a[pos] = i
            self.__gen(pos + 1, i, n, k)

    def generate_lexicographically(self, num_types, length):
        self.seqs = []
        self.a = np.zeros(length, dtype=int)
        self.__gen(0, 0, num_types, length)
        return np.copy(self.seqs)

    def __gen_with_gaps(self, pos, el, d, n, k):
        if pos == k:
            if self.is_generated:
                self.skipped += 1
                return
            self.is_generated = True
            self.seqs.append(np.copy(self.a))
            return
        for i in range(el, n):
            if self.is_generated:
                left_on_suffix = CombinationsWithRepetitions.number(n - i, k - pos - 1)
                if d - 1 - self.skipped >= left_on_suffix:
                    self.skipped += left_on_suffix
                    continue
            if d - 1 == self.skipped:
                self.is_generated = False
                self.skipped = 0
            self.a[pos] = i
            self.__gen_with_gaps(pos + 1, i, d, n, k)

    def generate_lexicographically_with_gaps(self, num_types, length, gap):
        self.seqs = []
        self.a = np.zeros(length, dtype=int)
        self.skipped = 0
        self.is_generated = False
        self.__gen_with_gaps(0, 0, gap, num_types, length)
        return np.copy(self.seqs)

    @staticmethod
    def number(num_types, length):
        return math.comb(num_types + length - 1, length)


class GeneratorUniform(GeneratorHarmonic):
    def __init__(self, filter_dist_matrix, d1, seqs, offset):
        super().__init__(filter_dist_matrix, d1, seqs, offset)

    def generate_distant_offspring(self, x, d):
        pos = np.random.randint(1, len(self.seqs))
        offspring = self._to_offspring(x, self.seqs[pos])
        self.dist_from_x = self.d1(x, offspring)
        return offspring


class Generator(AbstractGenerator):
    def __init__(self, filter_dist_matrix, d1):
        super().__init__(d1)
        self.matrix = filter_dist_matrix
        self.d1 = d1

    def __construct_individual(self, proportions):
        y = np.zeros(len(self.x), dtype=int)
        for i in range(len(proportions)):
            ind = int(len(self.matrix[self.x[i]]) * proportions[i])
            _, value = self.sorted_rows[i][ind]
            y[i] = value
        return y

    def generate_distant_offspring(self, x, d):
        self.x = x
        self.sorted_rows = []
        for i in range(len(x)):
            tmp = [(self.matrix[x[i]][j], j) for j in range(len(self.matrix[x[i]]))]
            tmp.sort()
            self.sorted_rows.append(tmp)

        def my_obj_func(proportions):
            y = self.__construct_individual(proportions)
            dist = self.d1(x, y)
            return abs(d - dist)

        res = cma.fmin(my_obj_func, np.random.uniform(0, 1., size=len(x)), 1.,
                       options={'bounds': [[0.] * len(x), [1.] * len(x)], 'maxfevals': 100, 'seed': 0})
        print(res[1])
        return self.__construct_individual(res[0])


def create_offspring_generator(inst, d0_method, d1_method, generating_method, seqs=None, offset=None, budget=None):
    dist_matrix = FilterDistanceFactory(inst) \
        .create_precomputed_filter_distance_matrix(d0_method, f'precomputedFiltersDists/method{d0_method}.txt')

    def d0(s1, s2):
        return dist_matrix[int(s1), int(s2)]

    d1 = SequenceDistanceFactory(d0).create_sequence_distance(d1_method)
    if generating_method == 'cma':
        return Generator(dist_matrix, d1)
    if generating_method == 1:
        return Generator1(dist_matrix, d1)
    if generating_method == 2:
        return Generator2(dist_matrix, d1)
    if generating_method == 'harmonic':
        return GeneratorHarmonic(dist_matrix, d1, seqs, offset)
    if generating_method == 'uniform':
        return GeneratorUniform(dist_matrix, d1, seqs, offset)
    if generating_method == 'ea':
        return GeneratorEA(dist_matrix, d1, budget)
