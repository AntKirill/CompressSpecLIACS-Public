import functools

import numpy as np
import utils
import algorithms


def run_es():
    inst = utils.create_instrument()
    M = 640  # number of filters in the sequeance of filters
    R = 16  # reduced dimensionality
    mu_, lambda_ = 30, 30
    lib_size = inst.filterlibrarysize

    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)

    seqs = algorithms.CombinationsWithRepetitions() \
        .generate_lexicographically_with_gaps(inst.filterlibrarysize, 16, 10 ** 42)
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'harmonic', seqs=seqs, offset=0)

    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-myes-harmonic', 'myes', 
f'({mu_}+{lambda_}) es with harmonic mutations. Combinations with repetitions are generated for all the filters ({lib_size} filters) and {R} segments with gap=10**42. Objective function is defined on {R} segments', generator)
    f = utils.add_segm_dim_reduction(f, M, R)

    alg = algorithms.MyES(None, M, 20000, mu_, lambda_, 0., generator)
    pop = [np.random.randint(0, lib_size, R) for _ in range(mu_)]

    alg(f, pop)


def run_rls():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-rls', 'rls', 'rls in subspaces of the same filters')
    my_mutation = functools.partial(algorithms.uniform_mutation, r=L)
    alg = algorithms.RLSSubspaces(my_mutation, 100, M, L, 0)
    initial = utils.read_selection('designs/filterguess-sron')
    alg(f, initial)


def run_ea():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-ea', 'ea', '(1+1) EA')
    my_mutation = functools.partial(algorithms.uniform_mutation, r=L)
    alg = algorithms.EA(my_mutation, 10000, M, L, 0)
    initial = utils.read_selection('designs/filterguess-sron')
    alg(f, initial)


def run_bo():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-bo', 'bo', 'bo')
    f = utils.add_segm_dim_reduction(f, M, 16)
    alg = algorithms.NoiseBOWrapper(50, 1000, 16, L, 0)
    alg(f)


def run_mies():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, 'experiments-mies', 'mies', 'mies')
    f = utils.add_segm_dim_reduction(f, M, 16)
    alg = algorithms.MIESWrapper(10000, 16, L, 0)
    alg(f)


def main():
    run_es()


if __name__ == '__main__':
    main()
