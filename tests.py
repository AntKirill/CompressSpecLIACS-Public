import functools
import json
import timeit

import numpy as np
import pytest
import random

import algorithms
import bootstrap
import utils


def test_dist():
    inst = utils.create_instrument()
    lib_size = inst.filterlibrarysize
    eps = 0.000001
    random.seed(0)
    for method in [2, 3]:
        d0 = utils.FilterDistanceFactory(inst).create_filters_distance(method)
        for _ in range(2000):
            f1, f2, f3 = random.sample(range(lib_size), 3)
            assert d0(f1, f2) > 0.
            assert d0(f1, f1) == pytest.approx(0., eps)
            assert d0(f1, f2) == pytest.approx(d0(f2, f1), eps)
            assert d0(f1, f2) + d0(f2, f3) > d0(f1, f3) - eps


def test_sequence_dist():
    inst = utils.create_instrument()
    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(d0).create_sequence_distance('kirill')
    seq1 = utils.read_selection('designs/newDesign')
    seq2 = utils.read_selection('designs/dNoSegmInit')
    dist = d1(seq1, seq2)
    permutation = d1.get_permutation(seq1, seq2)
    dist_perm = sum(d0(seq1[i], seq2[permutation[i]]) for i in range(len(permutation)))
    assert dist == pytest.approx(dist_perm, 0.0001)
    print(dist)


def test_obj_function_time():
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    f1 = utils.ObjFunctionPure(instrument, constants)
    f2 = utils.ObjFunctionAverageSquare(instrument, constants, 2000)
    seq = utils.read_selection('designs/newDesign')

    n = 100
    timer = timeit.Timer(functools.partial(f1, seq))
    result = timer.timeit(n)
    print(f"Execution f1 time is {result / n} seconds")

    n = 5
    timer = timeit.Timer(functools.partial(f2, seq))
    result = timer.timeit(n)
    print(f"Execution f2 time is {result / n} seconds")


def test_transmission_profiles():
    instrument = utils.create_instrument()
    selection = utils.read_selection('designs/newDesign')
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    vis = utils.SequenceFiltersVisualization(instrument, constants)
    vis.save_transmission_profiles(selection, 'tp.pdf', (20, 32))


def test_transmission_profiles_2():
    inst = utils.create_instrument()
    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(d0).create_sequence_distance('kirill')
    dim_reduction = utils.SegmentsDimReduction(640, 16)
    selection1 = utils.read_selection('designs/newDesign')
    selection1_reduced = dim_reduction.to_reduced(selection1)
    selection2_reduced = utils.read_selection('designs/dSegmGood')
    selection2 = dim_reduction.to_original(selection2_reduced)
    dist = d1(selection1, selection2)
    print(dist)
    permutation = d1.get_permutation(selection1_reduced, selection2_reduced)
    selection2_reduced = [selection2_reduced[permutation[i]] for i in range(len(permutation))]
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    vis = utils.SegmentedSequenceFiltersVisualization(inst, constants, dim_reduction)
    vis.save_transmission_profiles(selection1_reduced, 'tp1.pdf', (4, 4))
    vis.save_transmission_profiles(selection2_reduced, 'tp2.pdf', (4, 4))


def test_logs():
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    f = utils.ObjFunctionAverageSquare(instrument, constants, 1000)
    logger_f = utils.add_logger(f, 640, '_tmp_test', 'test', 'test')
    wrapped_f = utils.add_segm_dim_reduction(logger_f, 640, 16)
    selection = utils.read_selection('designs/dSegmGood')
    N = 5
    for i in range(N):
        f_obj = wrapped_f(selection)
        print(f_obj)


def test_dim_reduction():
    dim_reduction = utils.SegmentsDimReduction(640, 16)
    selection = utils.read_selection('designs/dSegmGood')
    original = dim_reduction.to_original(selection)
    print()
    print(*original, sep=' ')


def do_test_generator(generator):
    selection = utils.read_selection('designs/newDesign')
    timer = timeit.Timer(functools.partial(generator.generate_distant_offspring, selection, 0.5))
    result = timer.timeit(1)
    print(f"Execution f1 time is {result / 1} seconds")
    generator.generate_distant_offspring(selection, 1.)
    generator.generate_distant_offspring(selection, 0.5)
    generator.generate_distant_offspring(selection, 0.2)
    generator.generate_distant_offspring(selection, 0.1)
    generator.generate_distant_offspring(selection, 0.01)


def test_generator_1():
    inst = utils.create_instrument()
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 1)
    do_test_generator(generator)


def test_generator_2():
    inst = utils.create_instrument()
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 2)
    do_test_generator(generator)


def test_algorithms_MyES():
    inst = utils.create_instrument()
    M = 640
    lib_size = inst.filterlibrarysize
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 2)
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, '_tmp_test', 'alg_name', 'alg_info')
    f = utils.add_segm_dim_reduction(f, M, 16)

    mu_, lambda_ = 2, 2
    alg = algorithms.MyES(None, M, 1, mu_, lambda_, 0.1, generator)
    pop = [np.random.randint(0, lib_size, 16) for _ in range(mu_)]
    alg(f, initial=pop)


def test_algorithms_MyES1():
    inst = utils.create_instrument()
    M = 640
    lib_size = inst.filterlibrarysize
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 2)
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, '_tmp_test', 'alg_name', 'alg_info')
    f = utils.add_segm_dim_reduction(f, M, 16)

    mu_, lambda_ = 2, 2
    alg = algorithms.MyES1(None, M, 1, mu_, lambda_, 0.1, generator)
    pop = [np.random.randint(0, lib_size, 16) for _ in range(mu_)]
    alg(f, initial=pop)


def test_rls():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, '_tmp_test', 'alg_name', 'alg_info')
    my_mutation = functools.partial(algorithms.uniform_mutation, r=L)
    alg = algorithms.RLSSubspaces(my_mutation, 5, M, L, 0)
    initial = np.zeros(M, dtype=int)
    alg(f, initial)


def test_ea():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, '_tmp_test', 'alg_name', 'alg_info')
    my_mutation = functools.partial(algorithms.uniform_mutation, r=L)
    alg = algorithms.EA(my_mutation, 5, M, L, 0)
    initial = inst.filterguess()
    alg(f, initial)


def test_bo():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, '_tmp_test', 'alg_name', 'alg_info')
    f = utils.add_segm_dim_reduction(f, M, 16)
    alg = algorithms.NoiseBOWrapper(5, 2, 16, L, 0)
    alg(f)


def test_mies():
    inst = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    M, L = 640, inst.filterlibrarysize
    f = utils.ObjFunctionAverageSquare(inst, constants)
    f = utils.add_logger(f, M, '_tmp_test', 'alg_name', 'alg_info')
    f = utils.add_segm_dim_reduction(f, M, 16)
    alg = algorithms.MIESWrapper(5, 16, L, 0)
    alg(f)


def test_rls_configs():
    json_config = """{
    "algorithm": "RLS",
    "constants": {"nCH4": 1500, "albedo": 0.15, "sza": 70},
    "budget": 1,
    "genotype_mutation_operator": "uniform",
    "dim": 640,
    "initial_population": [729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,729,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,3108,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2990,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,3982,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,2996,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,3313,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,3047,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,2966,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,1832,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,699,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782,1782],
    "logger_root": "_tmp_experiments-rls",
    "algorithm_info": "rls"
    }"""
    json_obj = json.loads(json_config)
    config = bootstrap.ExperimentConfig(**json_obj)
    alg = bootstrap.Experiment().create_experiment(config)
    alg()


def test_bo_configs():
    json_config = """{
    "algorithm": "BO",
    "constants": {"nCH4": 1500, "albedo": 0.15, "sza": 70},
    "budget": 7,
    "doe_size": 5,
    "initial_population": [1],
    "dim": 640,
    "logger_root": "_tmp_experiments-bo",
    "algorithm_info": "bo"
    }"""
    json_obj = json.loads(json_config)
    config = bootstrap.ExperimentConfig(**json_obj)
    alg = bootstrap.Experiment().create_experiment(config)
    alg()

