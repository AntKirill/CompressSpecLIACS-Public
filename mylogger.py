import os
import json
from typing import List

import numpy as np


class LoggingBestSoFar:
    def __init__(self):
        self.min_dist = float("inf")

    def needToLog(self, value):
        if value < self.min_dist:
            self.min_dist = value
            return True
        return False


class LoggingAll:
    def needToLog(self, value):
        return True


class MyLogger:
    def __init__(self, root='experiments', folder_name='all', algorithm_name='UNKNOWN', suite='unkown suite', algorithm_info='algorithm_info', logStrategy=LoggingAll, isLogArg=False, verbose=True, instance=None):
        self.root = root
        self.folder_name = MyLogger.__generate_dir_name(f'{root}/{folder_name}', instance)
        self.algorithm_name = algorithm_name
        self.algorithm_info = algorithm_info
        self.suite = suite
        self.meta = {'version': "0.3.5",
                     'suite': suite,
                     'algorithm': {'name': algorithm_name, 'info': algorithm_info},
                     'attributes': ['evaluations', 'raw_y'],
                     'scenarios': []}
        self.myLogStrategy = logStrategy()
        self.isLogArg = isLogArg
        self.verbose = verbose
        self.instance = None
        self.meta_full_path = None
        self.meta_path = None
        self.log_file_full_path = None
        self.log_file_path = None
        self.all_extra_info_getters = []
        self.algorithms = []
        self.problem_dim = None

    @staticmethod
    def __generate_dir_name(name, x):
        dir_name = (name + ('-' + str(x))).strip()
        os.makedirs(dir_name, exist_ok=False)
        return dir_name

    def watch(self, algorithm, extra_data: List):
        self.algorithms.append(algorithm)
        self.all_extra_info_getters.append(extra_data)
        for extra_info in extra_data:
            self.meta['attributes'].append(extra_info)

    def set_up_logger_for_problem(self, function_id, function_name, is_maximization, instance, dim):
        self.log_file_path = f'data_f{function_id}_{function_name}/IOHprofiler_f{function_id}_DIM{dim}.dat'
        self.log_file_full_path = f'{self.folder_name}/{self.log_file_path}'
        self.meta_path = f'IOHprofiler_f{function_id}_{function_name}.json'
        self.meta_full_path = f'{self.folder_name}/{self.meta_path}'
        self.meta['function_id'] = function_id
        self.meta['function_name'] = function_name
        self.meta['maximization'] = is_maximization
        self.instance = instance
        self.problem_dim = dim
        self.meta['scenarios'].append({'dimension': dim, 'path': self.log_file_path, 'runs': [
                                      {'instance': instance, 'evals': 0, 'best': {}}]})
        os.makedirs(os.path.dirname(self.log_file_full_path), exist_ok=True)

    def log_config(self, config):
       with open(f'{self.folder_name}/config.json', 'w') as f:
           f.write(config.to_json())

    def log_column_names(self):
        with open(self.log_file_full_path, 'w') as f:
            f.write('evaluations raw_y')
            for extra_info_getter in self.all_extra_info_getters:
                for extra_info in extra_info_getter:
                    f.write(f' {extra_info}')
            if self.isLogArg:
                for i in range(self.problem_dim):
                    f.write(f' x{i}')
            f.write('\n')

    def log_eval(self, evaluation_number, arg, value):
        if not os.path.exists(self.log_file_full_path):
            self.log_column_names()
            if self.verbose:
                print(f'Logging to {self.folder_name}')
        with open(self.log_file_full_path, 'a') as f:
            f.write(f'{evaluation_number} {value}')
            for i in range(len(self.all_extra_info_getters)):
                for fu in self.all_extra_info_getters[i]:
                    try:
                        extra_info = getattr(self.algorithms[i], fu)
                    except Exception:
                        extra_info = 'None'
                    f.write(f' {extra_info}')
            if self.isLogArg:
                for element in arg:
                    f.write(f' {element}')
            f.write('\n')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(MyLogger.NpEncoder, self).default(obj)

    def log_meta(self, evaluation_number, arg, value, eval_min, arg_min, min_dist):
        self.meta['scenarios'][0]['runs'][0] = {
            'instance': self.instance,
            'evals': evaluation_number,
            'best': {'evals': eval_min, 'y': min_dist, 'x': list(arg_min)}}
        with open(self.meta_full_path, 'w') as f:
            json.dump(self.meta, f, indent=4, cls=MyLogger.NpEncoder)

    def log(self, evaluation_number, arg, value, eval_min, arg_min_dist, min_dist):
        if self.myLogStrategy.needToLog(value):
            self.log_eval(evaluation_number, arg, value)
            self.log_meta(evaluation_number, arg, value,
                          eval_min, arg_min_dist, min_dist)


class MyObjectiveFunctionWrapper:
    def __init__(self, f, dim, fname, fid=25, optimum=0, isMaximization=False, instance=0):
        self.my_loggers = []
        self.my_function = f
        self.dim = dim
        self.optimum = optimum
        self.fname = fname
        self.isMaximization = isMaximization
        self.instance = instance
        self.fid = fid
        self.cnt_eval = 0
        self.eval_min = -1
        self.min_distance = float('inf')
        self.arg_min = None

    def __call__(self, x, *args, **kwargs):
        cur_value = self.my_function(x, *args, **kwargs)
        self.cnt_eval += 1
        distance = cur_value - self.optimum
        if distance < self.min_distance:
            self.min_distance = distance
            self.arg_min = x
            self.eval_min = self.cnt_eval
            print('New min:', self.min_distance, flush=True)
        for l in self.my_loggers:
            l.log(self.cnt_eval, x, distance, self.eval_min,
                  self.arg_min, self.min_distance)
        return cur_value

    def attach_logger(self, logger):
        self.my_loggers.append(logger)
        logger.set_up_logger_for_problem(
            self.fid, self.fname, self.isMaximization, self.instance, self.dim)
