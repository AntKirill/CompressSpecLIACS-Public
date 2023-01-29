import functools
import json
from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np

import algorithms
import utils
from instrumentsimulator import InstrumentSimulator


@dataclass
class ExperimentConfig:
    algorithm: str
    constants: dict
    budget: int = 1000
    dim_reduction_method: str = None
    segments_number: int = None
    mu_: int = None
    lambda_: int = None
    generator_type: int = None
    initial_distance: float = None
    d0_method: int = None
    d1_method: int = None
    genotype_mutation_operator: str = None
    dim: int = 640
    initial_population: List = None  # This can be List of ints - individual, list of individuals or list of files
    logger_root: str = 'experiments'
    algorithm_info: str = 'alg_info'
    doe_size: int = None


@dataclass
class Experiment:
    instrument: InstrumentSimulator = None
    constants = None
    obj_function = None
    generator = None
    runnable_alg = None
    initial_pop = None
    genotype_mutation_operator = None

    def __create_instrument(self, config):
        self.instrument = utils.create_instrument()

    def __create_obj_function(self, config: ExperimentConfig):
        self.constants = utils.SRONConstants(**config.constants)
        self.obj_function = utils.ObjFunctionAverageSquare(self.instrument, self.constants)
        self.obj_function = utils.add_logger(self.obj_function, config.dim, config.logger_root, config.algorithm, config.algorithm_info)
        if config.dim_reduction_method == 'segments':
            self.obj_function = utils.add_segm_dim_reduction(self.obj_function, config.dim, config.segments_number)

    def __create_generator(self, config: ExperimentConfig):
        if config.generator_type is None:
            return
        generating_method = 'cma' if config.generator_type == 0 else config.generator_type
        self.generator = algorithms.create_offspring_generator(self.instrument, config.d0_method, config.d1_method,
                                                               generating_method)

    def __create_initial_population(self, config: ExperimentConfig):
        lib_size = self.instrument.filterlibrarysize
        if config.initial_population is None:
            self.initial_pop = [np.random.randint(0, lib_size, 16) for _ in range(config.mu_)]
        elif type(config.initial_population) is List and type(config.initial_population[0]) is str:
            self.initial_pop = [utils.read_selection(file_name) for file_name in config.initial_population]
        else:
            self.initial_pop = config.initial_population

    def __create_genotype_mutation_operator(self, config: ExperimentConfig):
        L = self.instrument.filterlibrarysize
        if config.genotype_mutation_operator == 'uniform':
            self.genotype_mutation_operator = functools.partial(algorithms.uniform_mutation, r=L)

    def __create_runnable_experiment(self, config: ExperimentConfig):
        L = self.instrument.filterlibrarysize
        if config.algorithm == 'MyES':
            alg = algorithms.MyES(None, config.dim, config.budget, config.mu_, config.lambda_, config.initial_distance,
                                  self.generator)
            self.runnable_alg = partial(alg, self.obj_function, self.initial_pop)
        if config.algorithm == 'MyES1':
            alg = algorithms.MyES1(None, config.dim, config.budget, config.mu_, config.lambda_, config.initial_distance,
                                   self.generator)
            self.runnable_alg = partial(alg, self.obj_function, self.initial_pop)
        if config.algorithm == 'RLS':
            alg = algorithms.RLSSubspaces(self.genotype_mutation_operator, config.budget, config.dim, L, 0)
            self.runnable_alg = partial(alg, self.obj_function, self.initial_pop)
        if config.algorithm == 'EA':
            alg = algorithms.EA(self.genotype_mutation_operator, config.budget, config.dim, L, 0)
            self.runnable_alg = partial(alg, self.obj_function, self.initial_pop)
        if config.algorithm == 'BO':
            alg = algorithms.NoiseBOWrapper(config.doe_size, config.budget, config.dim, L, 0)
            self.runnable_alg = partial(alg, self.obj_function)
        if config.algorithm == 'MIES':
            alg = algorithms.MIESWrapper(config.budget, config.dim, L, 0)
            self.runnable_alg = partial(alg, self.obj_function)

    def create_experiment(self, config):
        self.__create_instrument(config)
        self.__create_obj_function(config)
        self.__create_generator(config)
        self.__create_initial_population(config)
        self.__create_genotype_mutation_operator(config)
        self.__create_runnable_experiment(config)
        return self.runnable_alg


def parse_config(json_file):
    with open(json_file, 'r') as f:
        json_obj = json.load(f)
        return ExperimentConfig(**json_obj)


def run_experiment(json_file):
    config = parse_config(json_file)
    exp = Experiment()
    alg = exp.create_experiment(config)
    alg()


if __name__ == '__main__':
    run_experiment('config.json')
