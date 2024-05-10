from utilsV3 import *
import argparse
from umda import umda_Zn_minimization


class Individual:
    def __init__(self, F, x, population_number=0):
        self.noisy_objf = F
        self.x = x
        self.objf_x_samples = np.zeros(0)
        self.population_number = population_number

    def add_samples(self, n):
        self.noisy_objf(self.x, n)
        samples = self.noisy_objf.get_measurements()
        self.objf_x_samples = np.concatenate([self.objf_x_samples, samples])
        return self.objf_x_samples
    
    def obj_value(self):
        return np.mean(self.objf_x_samples**2)


def ea_simple(F, config, n_segms, mu_, lambda_, is_term, init_pop=None, mut_rate=None):
    L_size = F.instrument.filterlibrarysize
    if not init_pop:
        init_pop = []
    initial = init_pop + [utils.generate_random_solution(n_segms, L_size) for _ in range(max(0, mu_ - len(init_pop)))]
    population = [Individual(F, ind) for ind in initial]
    for ind in population:
        ind.add_samples(config.n_reps)
    spent_budget = len(initial)
    if not mut_rate:
        mut_rate = 2./n_segms
    while not is_term(0, spent_budget, 0):
        next_population = []
        for _ in range(lambda_):
            parent_num = np.random.choice(len(population), 1, replace=False)
            parent = population[parent_num[0]]
            num_pos = RandomEngine.sample_Binomial(len(parent.x), mut_rate)
            if num_pos == 0:
                num_pos = 1
            poss = np.random.choice(len(parent.x), num_pos, replace=False)
            offspring_genotype = np.copy(parent.x)
            for pos in poss:
                offspring_genotype[pos] = np.random.randint(0, L_size)
            spent_budget += 1
            ind = Individual(F, offspring_genotype)
            ind.add_samples(config.n_reps)
            next_population.append(ind)
        all_population = np.concatenate((population, next_population)).tolist()
        all_population.sort(key=lambda ind: ind.obj_value())
        population = all_population[:mu_]
    population.sort(key=lambda ind: ind.obj_value())
    return population[0]


class AbstractFilterSequenceOptimization(ABC):
    def __init__(self, F, dist_matrix_sorted, dist, config):
        self.F = F
        self.seq_length = F.search_space_dim
        self.L = F.instrument.filterlibrarysize
        self.dist_matrix_sorted = dist_matrix_sorted
        self.dist = dist
        self.config = config
        self.diversity_measure = None
        self.init_n_reps = config.n_reps
        self.dim_reducer = utils.SegmentsDimReduction(F.of.search_space_dim, self.seq_length)
        self.mean = 0.1
        utils.logger.watch(self, ['Diversity', 'iteration'])

    def calculate_diversity(self, population):
        d = 0
        for ind1 in population:
            for ind2 in population:
                d += self.dist(ind1.x, ind2.x)
        return d / (2 * len(population))
    
    def log_population(self, population):
        pop_numbers, xs, values = [], [], []
        for ind in population:
            xs.append(self.dim_reducer.to_original(ind.x))
            values.append(ind.obj_value())
            pop_numbers.append(ind.population_number)
        utils.logger.log_population(pop_numbers, xs, values)

    def choose(self, population):
        return np.random.choice(population, 2, replace=False)

    def crossover(self, parent1, parent2, c=0.5):
        offspring = np.copy(parent1.x)
        for i in range(len(offspring)):
            rnd = np.random.uniform()
            if rnd < c:
                offspring[i] = parent2.x[i]
        return offspring

    def ddMutation(self, x):
        s = self.stepSizeD.sample()
        step_size = np.sqrt(-np.log(1 - s)) / self.gamma
        y = self.mutator.mutation(x, step_size)
        self.step_size_error = self.dist(x, y) - step_size
        return y

    def survival_selection(self, population, new_population, mu_):
        all_population = np.concatenate((population, new_population)).tolist()
        all_population.sort(key=lambda ind: ind.obj_value())
        return all_population[:mu_]
    
    def select_best(self, inds):
        min_, argmin_ = float("inf"), None
        for ind in inds:
            f = ind.obj_value()
            if f < min_:
                min_, argmin_ = f, ind
        return argmin_

    @property
    def Diversity(self):
        return self.diversity_measure
    
    def configure_step_size_distribution(self):
        self.stepSizeD = RandomEngine.TruncatedExponentialDistribution().build(self.mean, 1e-9)
        if self.config.dd_mutation == 'ea':
            self.mutator = DDMutationEA(self.dist, self.dist_matrix_sorted)
        else:
            self.mutator = DDMutationUMDA(self.dist)
        x = utils.generate_random_solution(self.seq_length, self.L)
        self.dMin = findExtremeDist(x, self.dist_matrix_sorted, self.dist, 'min', self.mutator, self.config.d0_method, self.config.d1_method)
        self.dMax = findExtremeDist(x, self.dist_matrix_sorted, self.dist, 'max', self.mutator, self.config.d0_method, self.config.d1_method)
        self.gamma = findDistGamma(100*self.dMin, self.dMax)

    @abstractmethod
    def __call__(self, initial=None):
        pass


class DDGA(AbstractFilterSequenceOptimization):
    def __call__(self, initial=None):
        """
        Minimization of objective function
        """
        self.configure_step_size_distribution()
        if initial is None:
            initial = [utils.generate_random_solution(self.seq_length, self.L) for _ in range(self.config.mu_)]
        self.population = [Individual(self.F, ind, None) for ind in initial]
        for ind in self.population:
            ind.add_samples(self.init_n_reps)
        generations_number = (self.config.budget - self.config.mu_) // self.config.lambda_
        for self.iteration in range(generations_number):
            self.diversity_measure = self.calculate_diversity(self.population)
            self.log_population(self.population)
            self.next_population = []
            for _ in range(self.config.lambda_):
                parent1, parent2 = self.choose(self.population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.ddMutation(offspring)
                offspring_ind = Individual(self.F, offspring, self.iteration)
                offspring_ind.add_samples(self.init_n_reps)
                self.next_population.append(offspring_ind)
            self.population = self.survival_selection(self.population, self.next_population, self.config.mu_)
        self.population.sort(key=lambda ind: ind.obj_value())
        self.log_population(self.population)
        return self.population[0]


class DDOPLL(AbstractFilterSequenceOptimization):
    def __call__(self, initial=None):
        self.configure_step_size_distribution()
        if initial is None:
            initial = utils.generate_random_solution(self.seq_length, self.L)
        parent_ind = Individual(self.F, initial, None)
        parent_ind.add_samples(self.config.n_reps)
        y = []
        rests_evals = self.config.budget
        population_number = 0
        while rests_evals > 0:
            for i in range(self.config.lambda_):
                offspring = self.ddMutation(parent_ind.x)
                offspring_ind = Individual(self.F, offspring, population_number)
                offspring_ind.add_samples(self.config.n_reps)
                y.append(offspring_ind)
                rests_evals -= 1
            x1_ind = self.select_best(y)
            y = []
            for i in range(self.config.lambda_):
                offspring = self.crossover(parent_ind, x1_ind)
                offspring_ind = Individual(self.F, offspring, population_number)
                offspring_ind.add_samples(self.config.n_reps)
                y.append(offspring_ind)
                rests_evals -= 1
            x2_ind = self.select_best(y)
            parent_ind = self.select_best([parent_ind, x2_ind])
            population_number += 1
        return parent_ind


def run_optimization(config: Config):
    config.validate()
    PFR = create_profiled_obj_fun_for_reduced_space(config)
    global logger
    utils.logger.log_config(config)
    if config.algorithm == 'dd-ga' or config.algorithm == 'dd-opll':
        dist_matrix = utils.create_dist_matrix(PFR, config.d0_method)
        dist_matrix_sorted = sort_dist_matrix(dist_matrix)
        dist = utils.create_distance(PFR, dist_matrix, config.d1_method)
        if config.algorithm == 'dd-ga':
            opt = DDGA(PFR, dist_matrix_sorted, dist, config)
        elif config.algorithm == 'dd-opll':
            opt = DDOPLL(PFR, dist_matrix_sorted, dist, config)
        opt()
    elif config.algorithm == 'umda':
        umda_Zn_minimization(config.n_segms, PFR.instrument.filterlibrarysize,
                             config.mu_, config.lambda_, PFR,
                             lambda i1, s, i2: s > config.budget, config.is_log_distr_umda)
    elif config.algorithm == 'ea-simple':
        solution = ea_simple(PFR, config, config.n_segms, config.mu_, config.lambda_, lambda i1, s, i2: s > config.budget)
        print(solution.obj_value)


def main():
    dc = Config()
    parser = argparse.ArgumentParser(description='Runs optimization of the function by SRON with configuration')
    parser.add_argument('-i', '--algorithm_info', help='Information of the optimization algorithm', default=dc.algorithm_info)
    parser.add_argument('-r', '--n_reps', help='Number of resampling per point', type=int, default=dc.n_reps)
    parser.add_argument('-m', '--mu', help='Number of parents in EA', type=int, default=dc.mu_)
    parser.add_argument('-l', '--lambda_', help='Number of offspring in EA', type=int, default=dc.lambda_)
    parser.add_argument('-d0', '--d0_method', help='Distance between filters', default=dc.d0_method)
    parser.add_argument('-d1', '--d1_method', help='Distance between sequences of filters', default=dc.d1_method)
    parser.add_argument('--mu_explore', help='Number of parents in UMDA during exploration', type=int, default=dc.mu_explore)
    parser.add_argument('--lambda_explore', help='Number of offspring in UMDA during exploration', type=int, default=dc.lambda_explore)
    parser.add_argument('--budget_explore', help='Max number of distance evals during exploration', type=int, default=dc.budget_explore)
    parser.add_argument('--mu_mutation', help='Number of parents in UMDA during mutation', type=int, default=dc.mu_mutation)
    parser.add_argument('--lambda_mutation', help='Number of offspring in UMDA during mutation', type=int, default=dc.lambda_mutation)
    parser.add_argument('--budget_mutation', help='Max number of distances evals during mutation', type=int, default=dc.budget_mutation)
    parser.add_argument('--seq_length', help='Target length of the sequence of filters', type=int, default=dc.seq_length)
    parser.add_argument('--log_distr', help='Flag to print distribution for every generation in UMDA', type=bool, default=dc.is_log_distr_umda)
    parser.add_argument('-o', '--dd_mutation', help='Distance-Driven mutation operator internal optimization', type=str, default=dc.dd_mutation,
                                choices=Config.supported_dd_mutations())
    parser.add_argument('-b', '--budget', help='Max number of obj function evals', type=int, default=dc.budget)
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-a', '--algorithm', help='Optimization algorithm', required=True,
                                choices=Config.implemented_algorithms())
    required_named.add_argument('-f', '--folder_name', help='Name of the folder with logs', required=True)
    required_named.add_argument('-n', '--n_segms', help='Number of segments', type=int, required=True)
    required_named.add_argument('--instance', help='Instance number', type=int, required=True)
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
               is_log_distr_umda=args.log_distr,
               dd_mutation=args.dd_mutation,
               instance=args.instance))
    

if __name__ == '__main__':
    main()
