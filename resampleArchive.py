import sys
import os
import pandas as pd
import numpy as np
import utils
import utilsV3


def process_row_k(F, filepath, kFrom, kTo):
    experiment_root = os.path.dirname(filepath)
    archive_resampling_dir = os.path.join(experiment_root, f'archiveResampling')
    os.makedirs(archive_resampling_dir, exist_ok=True)
    outputpath = os.path.join(archive_resampling_dir, f'samples_for_solutions_from_{kFrom}_to_{kTo}.csv')
    df = pd.read_csv(filepath, sep=' ', header=None)
    with open(outputpath, 'w') as file:
        for k in range(kFrom, kTo + 1):
            row = df.iloc[k].tolist()
            obj_value, design = row[0], [int(i) for i in row[1:]]
            obj_value1 = F(design, 10000)
            samples = F.get_measurements()
            print('Old', obj_value, 'New', obj_value1)
            print(*design, *samples, file=file)


def main(filepath, kFrom, kToIncl):
    D = utilsV3.CriteriaD()
    F = utilsV3.CriteriaF(D)
    process_row_k(F, filepath, int(kFrom), int(kToIncl))


if __name__ == '__main__':
    main(*sys.argv[1:])
