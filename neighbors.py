import os
import numpy as np
import myrandom
import utils
import objf
import argparse
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


rnd = myrandom.RandomEngine(1)
L_size = 4374


def cyclic_plus(x, d):
    y = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        if x[i] + d[i] < 0:
            y[i] = L_size - 1
        elif x[i] + d[i] >= L_size:
            y[i] = 0
        else:
            if d[i] > 0:
                y[i] = rnd.sample_int_uniform(x[i] + 1, L_size)
            elif d[i] < 0:
                y[i] = rnd.sample_int_uniform(0, x[i])
            else:
                y[i] = x[i]
    return y


def single_comp_changes_by_lex_order(n, p):
    ans = np.zeros(n, dtype=int)
    if p <= n:
        ans[p - 1] = -1
    else:
        ans[n - (p - n)] = 1
    return ans


# n is number of segments, k is number of neighbors
def generate_single_comp_changes(x_ref, n, k):
    sel = rnd.sample_combination_uniform(2*n, k)
    np.random.shuffle(sel)
    objs = [None] * k
    for i in range(k):
        delta = single_comp_changes_by_lex_order(n, sel[i])
        objs[i] = cyclic_plus(x_ref, delta)
    return objs


def test(n, k):
    np.random.seed(1)
    x_ref = np.random.randint(0, L_size, n)
    ns = generate_single_comp_changes(x_ref, n, k)
    print(x_ref.tolist())
    for nn in ns:
        assert np.linalg.norm(x_ref - nn) > 1e-9
        print(nn.tolist())


def experiment_hamming_genconfig(folder, n, k):
    x_ref = np.random.randint(0, L_size, n)
    xs = generate_single_comp_changes(x_ref, n, k)
    r = utils.SegmentsDimReduction(640, n)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(f'{folder}/{0}', 'w') as f:
        print(*r.to_original(x_ref), file=f)
        print(*r.to_original(x_ref), file=f)
    for i in range(len(xs)):
        with open(f'{folder}/{i+1}', 'w') as f:
            print(*r.to_original(x_ref), file=f)
            print(*r.to_original(xs[i]), file=f)


def experiment_hamming_run(config_file_name, results_file_name, N):
    with open(config_file_name, 'r') as f:
        x_ref = list(map(int, f.readline().split()))
        x = list(map(int, f.readline().split()))
    folder = os.path.dirname(results_file_name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    F = objf.ObjFunSRON(1)
    with open(results_file_name, 'w') as f:
        print(*x_ref, file=f)
        print(*x, file=f)
        for i in range(N):
            value = F(x)
            print(value, file=f)


def parse_result(result_file_name):
    values = []
    with open(result_file_name, 'r') as f:
        cnt = 0
        for line in f:
            if cnt == 0:
                cnt += 1
                x_ref = list(map(int, line.split()))
            elif cnt == 1:
                cnt += 1
                x = list(map(int, line.split()))
            else:
                values.append(float(line))
    return x_ref, x, values


def build_pdf(k, vals):
    fig = plt.figure()
    plt.hist(vals, bins=1000)
    fig.savefig(f'pdfs/pdf_{k}.pdf')
    plt.close()


def process_results(process_folder, k):
    d = []
    sims = []
    for i in range(0, k + 1):
        x_ref, x, values = parse_result(f'{process_folder}/{i}')
        cosine_sim = np.dot(x_ref, x) / np.linalg.norm(x_ref) / np.linalg.norm(x)
        build_pdf(i, values)
        t = "{:.11f}".format(cosine_sim)
        if t[0] == '1':
            num = '1.0'
        else:
            num = '..' + t[9:]
        sims.append(num)
        d.append(values)
    fig, ax = plt.subplots()
    ax.boxplot(d, notch=True, sym='')
    plt.yscale('log')
    plt.xticks(range(1, len(d) + 1), sims, size=3)
    plt.savefig('box-plot-neighbors-hamming.pdf')
    plt.close()


@dataclass_json
@dataclass
class Config:
    experiment: str = 'hamming'
    n_segms: int = 16
    k: int = 32 # number of combinations that we are ready to wait for
    n_reps: int = 1000 # number of resamples of every point
    config_id: str = '__default__'
    config_folder: str = 'generated-configs-hamming'
    results_folder: str = 'results-hamming'
    process_folder: str = '__default__'


def main():
    c = Config()
    parser = argparse.ArgumentParser(description='Run experiments with distance to the neighbors')
    parser.add_argument('-e', '--experiment', help='Which experiment to run', choices=['hamming', 'custom'], default=c.experiment)
    parser.add_argument('-n', '--n_segms', help='Number of segments', type=int, default=c.n_segms)
    parser.add_argument('-k', '--combs', help='Number of combinations that we are ready to wait for', type=int, default=c.k)
    parser.add_argument('-r', '--n_reps', help='Number of resampling per point', type=int, default=c.n_reps)
    parser.add_argument('-f', '--config_id', help='Id of a config in case we run', type=str, default=c.config_id)
    parser.add_argument('-p', '--process_folder', help='Folder with results to process', type=str, default=c.process_folder)
    required_named = parser.add_argument_group('Required Named Arguments')
    required_named.add_argument('-m', '--mode', help='Do we generate configs or run experiment?', choices=['run', 'generate', 'process'], required=True)
    args = parser.parse_args()
    c.experiment = args.experiment
    c.n_segms = args.n_segms
    c.k = args.combs
    c.n_reps = args.n_reps
    c.config_id = args.config_id
    c.process_folder = args.process_folder
    if args.mode == 'generate':
        experiment_hamming_genconfig(c.config_folder, c.n_segms, c.k)
    elif args.mode == 'run':
        c.results_folder = 'results-' + c.experiment
        c.config_folder = 'generated-configs-' + c.experiment
        if c.experiment == 'hamming':
            experiment_hamming_run(f'{c.config_folder}/{c.config_id}', f'{c.results_folder}/{c.config_id}', c.n_reps)
        else:
            raise ValueError('Not implemented yet')
    elif args.mode == 'process':
        process_results(c.process_folder, c.k)
    else:
        raise ValueError(f'Invalid mode {args.mode}')


if __name__ == '__main__':
    main()

