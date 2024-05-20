from utilsV3 import *
import argparse
from umda import umda_Zn_minimization
import mipego
import nevergrad as ng
import os


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
    population = [Individual(F, ind, None) for ind in initial]
    for ind in population:
        ind.add_samples(config.n_reps)
    spent_budget = len(initial)
    if not mut_rate:
        mut_rate = 2./n_segms
    population_number = 0
    while not is_term(0, spent_budget, 0):
        print('Best-so-far:', population[0].obj_value())
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
            ind = Individual(F, offspring_genotype, population_number)
            ind.add_samples(config.n_reps)
            next_population.append(ind)
        all_population = np.concatenate((population, next_population)).tolist()
        all_population.sort(key=lambda ind: ind.obj_value())
        population = all_population[:mu_]
        population_number += 1
    population.sort(key=lambda ind: ind.obj_value())
    return population[0]


def WelchTTestPvalue(ind1, ind2):
    return stats.ttest_ind(ind1.objf_x_samples**2, ind2.objf_x_samples**2, equal_var=False).pvalue


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
        self.MAXSAMPLES = config.max_samples
        self.MAXSIGNIFICANCE = config.significance
        self.SAMPLESINC = config.sample_inc
        if hasattr(utils, 'logger'):
            utils.logger.watch(self, ['Diversity', 'iteration'])

    def calculate_diversity(self, population):
        d = 0
        for ind1 in population:
            for ind2 in population:
                d += self.dist(ind1.x, ind2.x)
        return d / (2 * len(population))
    
    def log_population(self, population, cur_pop_number):
        if not hasattr(utils, 'logger'):
            return
        self.diversity_measure = self.calculate_diversity(self.population)
        pop_numbers, xs, values, sizes, pvalues = [], [], [], [], []
        for ind in population:
            xs.append(self.dim_reducer.to_original(ind.x))
            values.append(ind.obj_value())
            pop_numbers.append(ind.population_number)
            sizes.append(len(ind.objf_x_samples))
            pvalues.append(WelchTTestPvalue(ind, population[0]))
        utils.logger.log_population(cur_pop_number, pop_numbers, xs, values, sizes, self.diversity_measure, pvalues)

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

    def select_robust_best(self, inds):
        pos_best = 0
        while True:
            is_best = True
            for i in range(len(inds)):
                if i == pos_best:
                    continue
                while True:
                    pvalue = WelchTTestPvalue(inds[pos_best], inds[i])
                    if pvalue < self.MAXSIGNIFICANCE or pvalue > 1 - self.MAXSIGNIFICANCE:
                        break
                    to_add_pos = pos_best
                    if len(inds[pos_best].objf_x_samples) > len(inds[i].objf_x_samples):
                        to_add_pos = i
                    if len(inds[to_add_pos].objf_x_samples) < self.MAXSAMPLES:
                        inds[to_add_pos].add_samples(self.SAMPLESINC)
                    else:
                        break
                if inds[pos_best].obj_value() > inds[i].obj_value():
                    pos_best = i
                    is_best = False
            if is_best:
                break
        return pos_best
    
    def robust_survival_selection(self, population, new_population, mu_):
        all_population = np.concatenate((population, new_population)).tolist()
        robust_best_inds = []
        for i in range(mu_):
            pos_robust_best = self.select_robust_best(all_population)
            robust_best_inds.append(all_population[pos_robust_best])
            all_population.pop(pos_robust_best)
        return robust_best_inds

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
    def my_survival_selection(self):
        if self.config.robustness == 'fixed':
            return self.survival_selection(self.population, self.next_population, self.config.mu_)
        elif self.config.robustness == 'welch':
            return self.robust_survival_selection(self.population, self.next_population, self.config.mu_)
        else:
            raise ValueError(f'Robustness method {self.config.robustness} is not implemented')

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
            self.log_population(self.population, self.iteration)
            self.next_population = []
            for _ in range(self.config.lambda_):
                parent1, parent2 = self.choose(self.population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.ddMutation(offspring)
                offspring_ind = Individual(self.F, offspring, self.iteration)
                offspring_ind.add_samples(self.init_n_reps)
                self.next_population.append(offspring_ind)
            self.population = self.my_survival_selection()
        self.population.sort(key=lambda ind: ind.obj_value())
        self.log_population(self.population, self.iteration)
        return self.population[0]


class DDES(DDGA):
    def choose(self, population):
        selected = np.random.choice(population, 1, replace=False)
        return selected[0], None        

    def crossover(self, parent1, parent2, c=0.5):
        return np.copy(parent1.x)
        

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


class DDLocalSearch(AbstractFilterSequenceOptimization):
    def __call__(self, initial=None):
        self.configure_step_size_distribution()
        if initial is None:
            initial = utils.generate_random_solution(self.F.search_space_dim, self.F.instrument.filterlibrarysize)
        x = initial
        x_obj = self.F(x, self.config.n_reps)
        print('Current best:', x_obj)
        for iteration in range(self.config.budget):
            y = self.ddMutation(x)
            y_obj = self.F(y, self.config.n_reps)
            if x_obj > y_obj:
                x, x_obj = y, y_obj
            print('Current best:', x_obj)
        return x, x_obj


class EASimple(AbstractFilterSequenceOptimization):
    def my_crossover(self, population):
            parent_num = np.random.choice(len(population), 1, replace=False)
            return population[parent_num[0]].x

    def __call__(self, init_pop=None):
        L_size = self.F.instrument.filterlibrarysize
        if not init_pop:
            init_pop = []
        initial = init_pop + [utils.generate_random_solution(self.config.n_segms, L_size) for _ in range(max(0, self.config.mu_ - len(init_pop)))]
        population = [Individual(self.F, ind, None) for ind in initial]
        for i, ind in enumerate(population):
            ind.add_samples(self.config.n_reps)
            print(f'Obj value init {i}:', ind.obj_value())
        spent_budget = len(initial)
        mut_rate = self.config.mut_rate/self.config.n_segms
        population_number = 0
        while spent_budget < self.config.budget:
            print('Best-so-far:', population[0].obj_value())
            next_population = []
            for _ in range(self.config.lambda_):
                cparent = self.my_crossover(population)
                num_pos = RandomEngine.sample_Binomial(len(cparent), mut_rate)
                if num_pos == 0:
                    num_pos = 1
                poss = np.random.choice(len(cparent), num_pos, replace=False)
                offspring_genotype = np.copy(cparent)
                for pos in poss:
                    offspring_genotype[pos] = np.random.randint(0, L_size)
                spent_budget += 1
                ind = Individual(self.F, offspring_genotype, population_number)
                ind.add_samples(self.config.n_reps)
                next_population.append(ind)
            all_population = np.concatenate((population, next_population)).tolist()
            all_population.sort(key=lambda ind: ind.obj_value())
            population = all_population[:self.config.mu_]
            population_number += 1
        population.sort(key=lambda ind: ind.obj_value())
        return population[0]
    

class EASimpleWithCrossover(EASimple):
    def my_crossover(self, population):
        p1, p2 = super().choose(population)
        return super().crossover(p1, p2)
    

class BO(AbstractFilterSequenceOptimization):
    def __call__(self, initial=None):
        f = lambda x: self.F(x, self.config.n_reps)
        I = mipego.OrdinalSpace([0, self.L - 1]) * self.config.n_segms
        model = mipego.RandomForest(levels=I.levels)
        doe_size = 5
        opt = mipego.NoisyBO(
            search_space=I,
            obj_fun=f,
            model=model,
            max_FEs=self.config.budget,
            DoE_size=doe_size,  # the initial DoE size
            # eval_type='dict',
            acquisition_fun='MGFI',
            acquisition_par={'t': 2},
            n_job=2,  # number of processes
            n_point=2,  # number of the candidate solution proposed in each iteration
            verbose=True  # turn this off, if you prefer no output
        )
        xopt, fopt, stop_dict = opt.run()
        return xopt, fopt
    

class MIES(AbstractFilterSequenceOptimization):
    def __call__(self, initial=None):
        f = lambda x: self.F(x, self.config.n_reps)
        I = mipego.OrdinalSpace([0, self.L - 1]) * self.config.n_segms
        mies = mipego.optimizer.mies.MIES(
            search_space=I,
            obj_func=f,
            max_eval=self.config.budget,
            mu_=self.config.mu_,
            lambda_=self.config.lambda_,
            verbose=True)
        xopt, fopt, stop_dict = mies.optimize()
        return xopt, fopt


class NGOptWrapper(AbstractFilterSequenceOptimization):
    def create_optimizer(self, arg):
        return ng.optimizers.NGOpt(parametrization=arg, budget=self.config.budget)

    def __call__(self, initial=None):
        f = lambda x: self.F(x, self.config.n_reps)
        arg1 = ng.p.Choice([i for i in range(self.F.instrument.filterlibrarysize)], repetitions=self.config.n_segms)
        optimizer = self.create_optimizer(arg1)
        recommendation = optimizer.minimize(f)
        return recommendation
    

class FastGANevergrad(NGOptWrapper):
    def create_optimizer(self, arg):
        print('Fast GA')
        return ng.optimizers.DoubleFastGADiscreteOnePlusOne(parametrization=arg, budget=self.config.budget)


class PortfolioNevergrad(NGOptWrapper):
    def create_optimizer(self, arg):
        print('RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne')
        return ng.optimizers.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne(parametrization=arg, budget=self.config.budget)


class BONevergrad(NGOptWrapper):
    def create_optimizer(self, arg):
        print('BO-NG')
        return ng.optimizers.BayesOptimBO(parametrization=arg, budget=self.config.budget)


class UMDA(AbstractFilterSequenceOptimization):
    def log_distribution(self, p, gen_number):
        with open(os.path.join(utils.logger.folder_name, 'umda_distr.txt'), 'a') as f:
            print(f'Generation {gen_number}, sz x crd', file=f)
            for i in range(len(p)):
                print(*p[i], sep=' ', file=f)
            print('', flush=True, file=f)

    def sample_from_distribution(self, p):
        x = np.zeros(len(p), dtype=int)
        for j in range(len(p)):
            x[j] = RandomEngine.sample_discrete_dist(p[j])
        return x

    def update_distribution(self, p, pop):
        for i in range(len(p)):
            cnt = np.zeros(self.L, dtype=int)
            for j in range(self.config.mu_):
                cnt[pop[j].x[i]] += 1
            for j in range(self.L):
                p[i][j] = min(max(cnt[j] / self.config.mu_, self.lb), self.ub)

    def optimize(self):
        spent_budget = 0
        best_fitness = float("inf")
        sol = None
        gen_number = 1
        while True:
            if self.config.is_log_distr_umda:
                self.log_distribution(self.p, gen_number)
            if spent_budget >= self.config.budget:
                break
            pop = []
            for i in range(self.config.lambda_):
                x = self.sample_from_distribution(self.p)
                x_ind = Individual(self.F, x, gen_number)
                x_ind.add_samples(self.config.n_reps)
                spent_budget += 1
                pop.append(x_ind)
            pop.sort(key=lambda ind: ind.obj_value())
            if pop[0].obj_value() < best_fitness:
                sol = pop[0]
                best_fitness = pop[0].obj_value()
            print(best_fitness)
            self.update_distribution(self.p, pop)
            gen_number += 1
        return sol.x, sol.obj_value()
    
    def __call__(self, initial=None):
        self.p = np.full((self.config.n_segms, self.L), 1. / self.L)
        self.lb = 1 / ((self.L - 1) * self.config.n_segms)
        self.ub = 1. - self.lb
        return self.optimize()
    

class UMDA1(UMDA):
    def log_distribution(self, p, gen_number):
        with open(os.path.join(utils.logger.folder_name, 'umda_distr.txt'), 'a') as f:
            print(f'Generation {gen_number}, sz', file=f)
            print(*p, sep=' ', file=f, flush=True)

    def sample_from_distribution(self, p):
        x = np.zeros(self.config.n_segms, dtype=int)
        for j in range(len(x)):
            x[j] = RandomEngine.sample_discrete_dist(p)
        return x

    def update_distribution(self, p, pop):
        cnt = np.zeros(self.L, dtype=int)
        sz = len(pop[0].x)
        for i in range(sz):
            for j in range(self.config.mu_):
                cnt[pop[j].x[i]] += 1
        for j in range(self.L):
            p[j] = min(max(cnt[j] / self.config.mu_ / sz, self.lb), self.ub)

    def __call__(self, initial=None):
        self.p = np.full(self.L, 1./self.L)
        self.lb = 1 / ((self.L - 1) * self.config.n_segms)
        self.ub = 1. - self.lb        
        return self.optimize()


def run_optimization(config: Config):
    PFR = create_profiled_obj_fun_for_reduced_space(config)
    global logger
    utils.logger.log_config(config)
    if config.algorithm.startswith('dd-'):
        dist_matrix = utils.create_dist_matrix(PFR, config.d0_method)
        dist_matrix_sorted = sort_dist_matrix(dist_matrix)
        dist = utils.create_distance(PFR, dist_matrix, config.d1_method)
        if config.algorithm == 'dd-ga':
            opt = DDGA(PFR, dist_matrix_sorted, dist, config)
        elif config.algorithm == 'dd-opll':
            opt = DDOPLL(PFR, dist_matrix_sorted, dist, config)
        elif config.algorithm == 'dd-ls':
            opt = DDLocalSearch(PFR, dist_matrix_sorted, dist, config)
        elif config.algorithm == 'dd-es':
            opt = DDES(PFR, dist_matrix_sorted, dist, config)
        opt()
    elif config.algorithm == 'umda':
        opt = UMDA(PFR, None, None, config)
        opt()
    elif config.algorithm == 'umda1':
        opt = UMDA1(PFR, None, None, config)
        opt()
    elif config.algorithm == 'ea-simple':
        # solution = ea_simple(PFR, config, config.n_segms, config.mu_, config.lambda_, lambda i1, s, i2: s > config.budget)
        opt = EASimple(PFR, None, None, config)
        opt()
    elif config.algorithm == 'ea-simple-cross':
        opt = EASimpleWithCrossover(PFR, None, None, config)
        opt()
    elif config.algorithm == 'mies':
        opt = MIES(PFR, None, None, config)
        opt()
    elif config.algorithm == 'ngopt':
        opt = NGOptWrapper(PFR, None, None, config)
        opt()
    elif config.algorithm == 'fastga-ng':
        opt = FastGANevergrad(PFR, None, None, config)
        opt()
    elif config.algorithm == 'portfolio-ng':
        opt = PortfolioNevergrad(PFR, None, None, config)
        opt()
    elif config.algorithm == 'bo-ng':
        opt = BONevergrad(PFR, None, None, config)
        opt()


def main():
    config = Config()
    parser = argparse.ArgumentParser(description='Runs optimization of the function by SRON with configuration')
    config.add_cml_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    run_optimization(config)
    

if __name__ == '__main__':
    main()
