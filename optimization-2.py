import argparse

import utils
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from ddmutation import FilterSequenceDDMutation
from myrandom import RandomEngine
from objf import *
from umda import umda_Zn_minimization
from utils import generate_random_solution, Individual, create_dist_matrix, create_distance


def create_profiled_obj_fun_for_reduced_space(config):
    return ReducedDimObjFunSRON(config.n_segms, ProfiledObjFunSRON(ObjFunSRON(config.n_reps), config))


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
    arg, value = umda_Zn_minimization(dim, crd, mu_, lambda_, f, term)
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
        umda_Zn_minimization(config.n_segms, F.instrument.filterlibrarysize, config.mu_, config.lambda_, F,
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
