import functools
import json
import math
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


def test_landscape_visualization():
    instrument = utils.create_instrument()
    select = utils.read_selection('designs/best-design-by-27-01')
    R = 16
    M = 640
    generator = algorithms.create_offspring_generator(instrument, 2, 'kirill', 'harmonic')
    dim_reduction = utils.SegmentsDimReduction(M, R)
    select_reduced = dim_reduction.to_reduced(select)
    visualization = utils.LandscapeVisualization()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    f = utils.ObjFunctionAverageSquare(instrument, constants)
    f = utils.add_segm_dim_reduction(f, M, 16)
    generator.generate_distant_offspring(select_reduced, 0.0012)
    np.random.seed(0)
    choice = np.random.choice([i for i in range(len(generator.seqs))], 200, replace=False)
    # chosen = [np.array(generator._to_offspring(select_reduced, generator.seqs[choice[i]]), dtype=int) for i in
    #                   range(len(choice))]
    chosen = []
    pos = set()
    for i in range(200):
        offspring = generator.generate_distant_offspring(select_reduced, 0.0012)
        if generator.pos not in pos:
            pos.add(generator.pos)
            chosen.append(offspring)

    print(len(chosen))
    chosen.append(select_reduced)
    X = np.array(chosen)
    visualization.visualize(X, f, generator.d1)


def test_transmission_profiles_2():
    inst = utils.create_instrument()
    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(d0).create_sequence_distance('kirill')
    dim_reduction = utils.SegmentsDimReduction(640, 16)
    selection1 = utils.read_selection('designs/best-design-by-27-01')
    selection1_reduced = dim_reduction.to_reduced(selection1)
    selection2_reduced = utils.read_selection('designs/dSegmGood')
    selection2 = dim_reduction.to_original(selection2_reduced)
    dist = d1(selection1, selection2)
    print(dist)
    selection1_reduced.sort()
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


def test_logs_with_generator():
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically(5, 16)
    generator = algorithms.create_offspring_generator(instrument, 2, 'kirill', 'harmonic', seqs, 0)
    f = utils.ObjFunctionAverageSquare(instrument, constants, 1000)
    f = utils.add_logger(f, 640, '_tmp_test', 'test', 'test', generator)
    f = utils.add_segm_dim_reduction(f, 640, 16)
    alg = algorithms.MyES(None, 640, 6, 2, 2, 0., generator)
    pop = [np.random.randint(0, instrument.filterlibrarysize, 16) for _ in range(2)]
    alg(f, pop)


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
    selection = utils.read_selection('designs/newDesign')
    for _ in range(10):
        generator.generate_distant_offspring(selection, 0.1)
        print(generator.distance_from_parent)


def test_generator_harmonic():
    inst = utils.create_instrument()
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically(5, 16)
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'harmonic', seqs, 0)
    selection = utils.read_selection('designs/newDesign')
    dim_red = utils.SegmentsDimReduction(640, 16)
    reduced_selection = dim_red.to_reduced(selection)
    timer = timeit.Timer(functools.partial(generator.generate_distant_offspring, reduced_selection, 0.0012))
    n = 10
    result = timer.timeit(n)
    print(f"Execution time is {result / n} seconds")


def test_generator_harmonic_1():
    inst = utils.create_instrument()
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically_with_gaps(inst.filterlibrarysize, 16, 10**42)
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'harmonic', seqs, 0)
    selection = utils.read_selection('designs/newDesign')
    dim_red = utils.SegmentsDimReduction(640, 16)
    reduced_selection = dim_red.to_reduced(selection)
    timer = timeit.Timer(functools.partial(generator.generate_distant_offspring, reduced_selection, 0))
    n = 10
    result = timer.timeit(n)
    print(*generator.dists[generator.sorted_ids])
    print(f"Execution time is {result / n} seconds")


def test_neighborhood_size_generator_harmonic_1():
    inst = utils.create_instrument()
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically_with_gaps(inst.filterlibrarysize, 16, 10**42)
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'harmonic1', seqs, 0)
    for i in range(1000):
        offspring = np.random.randint(0, inst.filterlibrarysize, 16)
        generator.generate_distant_offspring(offspring, 0)
        print(generator.dists[generator.sorted_ids[len(seqs)-1]])


def test_generator_uniform():
    inst = utils.create_instrument()
    generator = algorithms.create_offspring_generator(inst, 2, 'kirill', 'uniform')
    selection = utils.read_selection('designs/newDesign')
    dim_red = utils.SegmentsDimReduction(640, 16)
    reduced_selection = dim_red.to_reduced(selection)
    timer = timeit.Timer(functools.partial(generator.generate_distant_offspring, reduced_selection, 0.0012))
    n = 10
    result = timer.timeit(n)
    print(f"Execution time is {result / n} seconds")


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
    alg = algorithms.MIESWrapper(5, 16, L, 0, 15, 30)
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


def do_test_combinations_with_repetitions(n, k):
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically(n, k)
    assert len(seqs) == math.comb(n + k - 1, k)
    assert len(seqs) == algorithms.CombinationsWithRepetitions.number(n, k)


def do_test_combinations_with_repetitions_with_gaps(n, k, gap):
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically_with_gaps(n, k, gap)
    number = algorithms.CombinationsWithRepetitions.number(n, k)
    correct_len = number // gap
    correct_len += 1 if number % gap != 0 else 0
    assert len(seqs) == correct_len


def test_combinations_with_repetitions():
    do_test_combinations_with_repetitions(5, 1)
    do_test_combinations_with_repetitions(5, 3)
    do_test_combinations_with_repetitions(5, 4)
    do_test_combinations_with_repetitions(10, 5)
    n, k = 5, 2
    seqs = algorithms.CombinationsWithRepetitions().generate_lexicographically_with_gaps(n, k, 3)
    correct_seqs = np.array([[0, 0], [0, 3], [1, 2], [2, 2], [3, 3]])
    assert (seqs == correct_seqs).all()
    do_test_combinations_with_repetitions_with_gaps(5, 3, 4)
    do_test_combinations_with_repetitions_with_gaps(20, 32, 4*10**10)
