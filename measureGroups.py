import utilsV3
import os
import sys


M = 1000

def get_stats(F, x):
    F(x, M)
    return F.get_measurements()


def main(experiment_root, id):
    filepath = os.path.join(experiment_root, f'group_{id}.csv')
    with open(filepath, 'r') as fileRead:
        with open(os.path.join(experiment_root, f'group_samples_{id}.csv'), 'w') as fileWrite:
            D = utilsV3.CriteriaD()
            F = utilsV3.CriteriaF(D)
            for line in fileRead:
                sp = line.split(' ')
                x = [int(i) for i in sp[1:]]
                samples = get_stats(F, x)
                print(sp[0], *x, *samples, file=fileWrite)    


if __name__ == '__main__':
    main(*sys.argv[1:])
