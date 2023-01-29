from abc import ABC, abstractmethod

import cma
import numpy as np

import mipego
from utils import FilterDistanceFactory, SequenceDistanceFactory


class Individual:
    def __init__(self, genotype, obj_value):
        self.genotype = genotype
        self.obj_value = obj_value


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
        I = mipego.OrdinalSpace([0, self.L-1]) * self.M
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
    def __init__(self, budget, M, L, seed):
        self.M = M
        self.L = L
        self.budget = budget
        np.random.seed(seed)

    def __call__(self, f):
        I = mipego.OrdinalSpace([0, self.L-1]) * self.M
        mies = mipego.optimizer.mies.MIES(
            search_space=I,
            obj_func=f,
            max_eval=self.budget)
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


class MyES1(MyES):
    def crossover(self, parent1, parent2):
        offspring = np.copy(parent1.genotype)
        return offspring


class Generator1:
    def __init__(self, filter_dist_matrix, d1):
        self.matrix = filter_dist_matrix
        self.d1 = d1
        self.matrix_sorted_rows = []
        for i in range(len(self.matrix)):
            tmp = [(self.matrix[i][j], j) for j in range(len(self.matrix[i]))]
            tmp.sort()
            self.matrix_sorted_rows.append(tmp)

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
        print(max_v)
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
        print(self.d1(y, x))
        return y


class Generator:
    def __init__(self, filter_dist_matrix, d1):
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


def create_offspring_generator(inst, d0_method, d1_method, generating_method):
    dist_matrix = FilterDistanceFactory(inst) \
        .create_precomputed_filter_distance_matrix(d0_method, f'precomputedFiltersDists/method{d0_method}.txt')
    d0 = lambda s1, s2: dist_matrix[s1, s2]
    d1 = SequenceDistanceFactory(d0).create_sequence_distance(d1_method)
    if generating_method == 'cma':
        return Generator(dist_matrix, d1)
    if generating_method == 1:
        return Generator1(dist_matrix, d1)
    if generating_method == 2:
        return Generator2(dist_matrix, d1)
