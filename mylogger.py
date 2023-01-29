import os
import json
import numpy as np

import mylogger


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
    def __init__(self, root='experiments', folder_name='all', algorithm_name='UNKNOWN', suite='unkown suite', algorithm_info='algorithm_info', logStrategy=LoggingAll, isLogArg=False):
        self.root = root
        self.folder_name = MyLogger.__generate_dir_name(
            f'{root}/{folder_name}')
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

    @staticmethod
    def __generate_dir_name(name, x=0):
        while True:
            dir_name = (name + ('-' + str(x))).strip()
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                return dir_name
            else:
                x = x + 1

    def watch(self, algorithm, extra_data):
        self.algorithm = algorithm
        self.extra_info_getters = extra_data

    def set_up_logger_for_problem(self, function_id, function_name, isMaximization, instance, dim):
        self.log_file_path = f'data_f{function_id}_{function_name}/IOHprofiler_f{function_id}_DIM{dim}.dat'
        self.log_file_full_path = f'{self.folder_name}/{self.log_file_path}'
        self.meta_path = f'IOHprofiler_f{function_id}_{function_name}.json'
        self.meta_full_path = f'{self.folder_name}/{self.meta_path}'
        self.meta['function_id'] = function_id
        self.meta['function_name'] = function_name
        self.meta['maximization'] = isMaximization
        self.instance = instance
        self.meta['scenarios'].append({'dimension': dim, 'path': self.log_file_path, 'runs': [
                                      {'instance': instance, 'evals': 0, 'best': {}}]})
        os.makedirs(os.path.dirname(self.log_file_full_path), exist_ok=True)
        for extra_info in self.extra_info_getters:
            self.meta['attributes'].append(extra_info)
        with open(self.log_file_full_path, 'a') as f:
            f.write('evaluations raw_y')
            for extra_info in self.extra_info_getters:
                f.write(f' {extra_info}')
            if self.isLogArg:
                for i in range(dim):
                    f.write(f' x{i}')
            f.write('\n')

    def log_eval(self, evaluation_number, arg, value):
        with open(self.log_file_full_path, 'a') as f:
            f.write(f'{evaluation_number} {value}')
            for fu in self.extra_info_getters:
                try:
                    extra_info = getattr(self.algorithm, fu)
                except Exception as e:
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
            return super(mylogger.MyLogger.NpEncoder, self).default(obj)

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

    def __call__(self, x):
        cur_value = self.my_function(x)
        self.cnt_eval += 1
        distance = cur_value - self.optimum
        if distance < self.min_distance:
            self.min_distance = distance
            self.arg_min = x
            self.eval_min = self.cnt_eval
        for l in self.my_loggers:
            l.log(self.cnt_eval, x, distance, self.eval_min,
                  self.arg_min, self.min_distance)
        return cur_value

    def attach_logger(self, logger):
        self.my_loggers.append(logger)
        logger.set_up_logger_for_problem(
            self.fid, self.fname, self.isMaximization, self.instance, self.dim)