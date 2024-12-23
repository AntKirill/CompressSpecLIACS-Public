import functools

import numpy as np
import utils
import algorithms


def get_constants():
    return utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)


def run_es():
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    R = 16  # reduced dimensionality
    mu_, lambda_ = 15, 30
    # mu_, lambda_ = 5, 5
    lib_size = inst.filterlibrarysize

    constants = get_constants()

    mutation_distance = 0.01
    # generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'ea', budget=1000)
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 2)

    f = utils.ObjFunctionAverageSquare(inst, constants)
    # f = utils.add_logger(f, M, 'experiments-myes-generatorEA', 'myes',
    #                      f'({mu_}+{lambda_}) es with generator EA. This generator uses (1+5) EA with harmonic mutations of 1 filter to generate offspring distante to {mutation_distance:.5f} from the parent. Objective function is defined on {R} segments',
    #                      generator)
    f = utils.add_logger(f, M, 'experiments-myes-generator2', 'myes',
                         f'({mu_}+{lambda_}) es with generator g2',
                         generator)
    f = utils.add_segm_dim_reduction(f, M, R)

    alg = algorithms.MyES(None, M, 20000, mu_, lambda_, mutation_distance, generator)
    pop = [np.random.randint(0, lib_size, R) for _ in range(mu_)]

    alg(f, pop)


def run_es_distr():
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    R = 32  # reduced dimensionality
    mu_, lambda_ = 15, 30
    # mu_, lambda_ = 5, 5
    lib_size = inst.filterlibrarysize

    constants = get_constants()

    mutation_distance = 0.02
    distr = algorithms.Exponential()
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'ea', budget=1000)

    suffix = utils.now_date_time_str()
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, f'experiments-myes-generatorEA_{suffix}', 'myes',
                         f'({mu_}+{lambda_}) es with generator EA. This generator uses (1+5) EA with harmonic mutations of 1 filter to generate offspring distante to the given distance from the parent. Sequence distance is obtained using method kirill. This distance is drawn from the {distr.__class__.__name__} distribution with max_dist choosen uniformly at random from 0, to 0.01. Objective function is defined on {R} segments. SRON constants are: nCH4={constants.nCH4}, sza={constants.sza}, albedo={constants.albedo}',
                         generator)
    f = utils.add_segm_dim_reduction(f, M, R)
    utils.logger.watch(generator,
                       ['distance_from_parent', 'target_distance_from_parent', 'hamming_distance_from_parent'])

    alg = algorithms.MyESFixedDistDistribution(None, M, 20000, mu_, lambda_, mutation_distance, generator, distr)
    pop = [np.random.randint(0, lib_size, R) for _ in range(mu_)]

    alg(f, pop)


def run_es_new_dist_distr():
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    R = 16  # reduced dimensionality
    mu_, lambda_ = 15, 30
    # mu_, lambda_ = 5, 5
    lib_size = inst.filterlibrarysize

    constants = get_constants()

    distr = algorithms.Uniform()
    # mutation_distance = 15.
    # d0_method = 3
    # d1_method = '3'
    mutation_distance = 0.8e-6
    d0_method = 2
    d1_method = '2'
    generator = algorithms.create_offspring_generator(inst, d0_method, d1_method, 'ea1', budget=1000, M=M, R=R)

    suffix = utils.now_date_time_str()
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-myes-generatorEA-new-dists', 'myes',
                         f'({mu_}+{lambda_}) es with generator EA. This generator uses (1+5) EA with harmonic mutations of 1 filter to generate offspring distante to the given distance from the parent. Distance between filters is computed using {d0_method}. Distance between sequences is computed using method {d1_method}. This distance is drawn from the {distr.__class__.__name__} distribution with max_dist choosen uniformly at random from 0, to {mutation_distance:.10f}. Objective function is defined on {R} segments. SRON constants are: nCH4={constants.nCH4}, sza={constants.sza}, albedo={constants.albedo}')
    f = utils.add_segm_dim_reduction(f, M, R)
    utils.logger.watch(generator,
                       ['distance_from_parent', 'target_distance_from_parent', 'hamming_distance_from_parent'])

    alg = algorithms.MyESFixedDistDistribution(None, M, 10000, mu_, lambda_, mutation_distance, generator, distr)
    pop = [np.random.randint(0, lib_size, R) for _ in range(mu_)]

    alg(f, pop)


def run(mu_: int, lambda_: int, mutation_distance: float, d0_method: int, d1_method: str, inner_opt: str, R: int,
        results_root_prefix: str):
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    lib_size = inst.filterlibrarysize

    constants = get_constants()

    distr = algorithms.Uniform()
    generator = algorithms.create_offspring_generator(inst, d0_method, d1_method, inner_opt, budget=1000, M=M, R=R)

    suffix = utils.now_date_time_str()
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, f'{results_root_prefix}_{suffix}', results_root_prefix,
                         f'({mu_}+{lambda_}) es with generator EA. This generator uses (1+5) EA with harmonic mutations of 1 filter to generate offspring distante to the given distance from the parent. Distance between filters is computed using {d0_method}. Distance between sequences is computed using method {d1_method}. This distance is drawn from the {distr.__class__.__name__} distribution with max_dist choosen uniformly at random from 0, to {mutation_distance:.10f}. Objective function is defined on {R} segments. SRON constants are: nCH4={constants.nCH4}, sza={constants.sza}, albedo={constants.albedo}')
    f = utils.add_segm_dim_reduction(f, M, R)
    utils.logger.watch(generator,
                       ['distance_from_parent', 'target_distance_from_parent', 'hamming_distance_from_parent'])

    alg = algorithms.MyESFixedDistDistribution(None, M, 10000, mu_, lambda_, mutation_distance, generator, distr)
    pop = [np.random.randint(0, lib_size, R) for _ in range(mu_)]

    alg(f, pop)


def run_local_search():
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    R = 32  # reduced dimensionality

    constants = get_constants()

    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'ea', budget=1000)
    distribution = algorithms.Uniform()
    max_dist = 0.001
    initial_design = utils.read_selection('ls-imporoved')

    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-ls', 'sa',
                         f'Local Search using permutation sequence distances based on method 2. {R} segments are used. For generation of the solution in the neighbourhood generatorEA is used. This distance is drawn from the {distribution.__class__.__name__} distribution with max_dist {max_dist}.')
    f = utils.add_segm_dim_reduction(f, M, R)
    dim_reduction = utils.SegmentsDimReduction(M, R)
    reduced_initial_design = dim_reduction.to_reduced(initial_design)

    alg = algorithms.PhLocalSearch(1000, generator, max_dist, distribution)
    utils.logger.watch(generator,
                       ['distance_from_parent', 'target_distance_from_parent', 'hamming_distance_from_parent'])

    alg(f, reduced_initial_design)


def run_sa():
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    R = 16  # reduced dimensionality
    lib_size = inst.filterlibrarysize

    constants = get_constants()

    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'ea', budget=1000)

    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-sa', 'sa',
                         'Simulated Annealing using permutation sequence distances based on method 2. For generation of the solution in the neighbourhood generatorEA is used')
    f = utils.add_segm_dim_reduction(f, M, R)

    alg = algorithms.FiltersPhenoSimulatedAnnealing(100, 0, 0.99, generator, 0.000001)
    utils.logger.watch(alg, ['temperature', 'current_solution_quality', 'last_update_prob'])
    utils.logger.watch(generator, ['distance_from_parent', 'target_distance_from_parent'])
    pop = utils.read_selection('designs/best-design-by-27-01')
    dim_reduction = utils.SegmentsDimReduction(M, R)
    reduced_pop = dim_reduction.to_reduced(pop)

    alg(f, reduced_pop)


def run_rls():
    inst = utils.create_instrument()
    constants = get_constants()
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-rls', 'rls', 'rls in subspaces of the same filters')
    my_mutation = functools.partial(algorithms.uniform_mutation, r=L)
    alg = algorithms.RLSSubspaces(my_mutation, 100, M, L, 0)
    initial = utils.read_selection('designs/filterguess-sron')
    alg(f, initial)


def run_ea():
    inst = utils.create_instrument()
    constants = get_constants()
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-ea', 'ea', '(1+1) EA')
    my_mutation = functools.partial(algorithms.uniform_mutation, r=L)
    alg = algorithms.EA(my_mutation, 10000, M, L, 0)
    initial = utils.read_selection('designs/filterguess-sron')
    alg(f, initial)


def run_bo():
    inst = utils.create_instrument()
    constants = get_constants()
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-bo', 'bo', 'bo')
    f = utils.add_segm_dim_reduction(f, M, 16)
    alg = algorithms.NoiseBOWrapper(50, 1000, 16, L, 0)
    alg(f)


def run_mies():
    inst = utils.create_instrument()
    constants = get_constants()
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-mies', 'mies', 'mies')
    f = utils.add_segm_dim_reduction(f, M, 16)
    alg = algorithms.MIESWrapper(10000, 16, L, 0)
    alg(f)


def main():
    import os, psutil
    process = psutil.Process(os.getpid())

    mu_, lambda_ = 30, 30
    run(mu_=mu_,
        lambda_=lambda_,
        mutation_distance=0.02,
        d0_method=2,
        d1_method='kirill',
        inner_opt='ea',
        R=32,
        results_root_prefix='mk_2_s32')
    # run(mu_=mu_,
    #     lambda_=lambda_,
    #     mutation_distance=1.6e-6,
    #     d0_method=2,
    #     d1_method='2',
    #     inner_opt='ea1',
    #     R=32,
    #     results_root_prefix='m2_s32')
    # run(mu_=mu_,
    #     lambda_=lambda_,
    #     mutation_distance=30.,
    #     d0_method=3,
    #     d1_method='3',
    #     inner_opt='ea1',
    #     R=32,
    #     results_root_prefix='m3_s32')
    #
    # run(mu_=mu_,
    #     lambda_=lambda_,
    #     mutation_distance=0.0025,
    #     d0_method=2,
    #     d1_method='kirill',
    #     inner_opt='ea',
    #     R=4,
    #     results_root_prefix='mk_2_s4')
    # run(mu_=mu_,
    #     lambda_=lambda_,
    #     mutation_distance=0.2e-6,
    #     d0_method=2,
    #     d1_method='2',
    #     inner_opt='ea1',
    #     R=4,
    #     results_root_prefix='m2_s4')
    # run(mu_=mu_,
    #     lambda_=lambda_,
    #     mutation_distance=4.,
    #     d0_method=3,
    #     d1_method='3',
    #     inner_opt='ea1',
    #     R=4,
    #     results_root_prefix='m3_s4')

    print(process.memory_info().rss / 1024 / 1024, 'MB')


if __name__ == '__main__':
    main()
