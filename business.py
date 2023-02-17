import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import utils
import measure_noise


def read_vals(file_name):
    with open(file_name, 'r') as f:
        return list(map(float, f.read().split()))


def build_box_plot(values_file):
    vals = read_vals(values_file)
    fig = plt.figure()
    plt.boxplot(x=vals, vert=False)
    plt.grid()
    fig.savefig(f'box-plot.pdf')
    plt.close()


def sron_guess(n=5):
    n = int(n)
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    selection = instrument.filterguess()
    f = utils.ObjFunctionAverageSquare(instrument, constants)
    rv = np.zeros(n)
    for i in range(n):
        obj = f(selection)
        rv[i] = f.sron_precision
        print(obj, f.sron_bias, f.sron_precision, flush=True)
    return rv


def try_design(design_file, n=5):
    n = int(n)
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    f = utils.ObjFunctionAverageSquare(instrument, constants)
    selection = utils.read_selection(design_file)
    rv = np.zeros(n)
    for i in range(n):
        obj = f(selection)
        rv[i] = f.sron_precision
        print(obj, f.sron_bias, f.sron_precision, flush=True)
    return rv


def precompute_dists(method):
    method = int(method)
    inst = utils.create_instrument()
    d0 = utils.FilterDistanceFactory(inst).create_filters_distance(method)
    with open(f'precomputedFiltersDists/method{method}.txt', 'w') as f:
        for i in range(inst.filterlibrarysize):
            for j in range(i + 1, inst.filterlibrarysize):
                print(i, j, d0(i, j), file=f)
    return 'Success'


def build_tp(design_file, R=16, n=4, m=4):
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    selection = utils.read_selection(design_file)
    M = 640
    R, n, m = int(R), int(n), int(m)
    if R < M:
        dim_reduction = utils.SegmentsDimReduction(M, R)
        selection = dim_reduction.to_reduced(selection)
        vis = utils.SegmentedSequenceFiltersVisualization(instrument, constants, dim_reduction)
    else:
        vis = utils.SequenceFiltersVisualization(instrument, constants)
    selection.sort()

    vis.save_transmission_profiles(selection, 'tp.pdf', (n, m))
    sz = len(set(selection))
    print(f'Number of different filters is {sz}')


def build_pdf(values_file):
    vals = read_vals(values_file)
    fig = plt.figure()
    plt.hist(vals, bins=100)
    fig.savefig('pdf.pdf')
    plt.close()


def generate_tex():
    with open('template_tex', 'r') as f:
        template_str = f.read()
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32,
           33, 36, 37, 42, 47]
    with open('generated.tex', 'w') as f:
        for i in ids:
            s = template_str.replace("ID", str(i))
            print(s, end='\n\n', file=f, flush=True)


def structural_changes(bestsofarfile):
    inst = utils.create_instrument()

    @dataclass
    class MyEval:
        iteration: int
        obj: float
        sron_mean: float
        sron_precision: float
        design: List

    data = []
    with open(bestsofarfile, 'r') as f:
        r = csv.reader(f, delimiter=' ')
        next(r, None)
        for row in r:
            data.append(MyEval(int(row[0]), float(row[1]), float(row[2]), float(row[3]), [int(i) for i in row[4:]]))

    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(d0).create_sequence_distance('kirill')
    dists = np.zeros(len(data) - 1)
    for i in range(1, len(data)):
        dists[i-1] = d1(data[0].design, data[i].design)
    return dists


def extract_bestsofar(everyeval_path, budget):
    if not os.path.exists(everyeval_path):
        return
    with open(everyeval_path, 'r') as f:
        r = csv.reader(f, delimiter=' ')
        next(r, None)
        global_min = float("inf")
        x = []
        y = []
        cnt = 0
        for row in r:
            if row[3] == 'sron_precision':
                continue
            v = float(row[3])
            if v < global_min:
                x.append(cnt)
                y.append(v)
                global_min = v
            else:
                x.append(cnt)
                y.append(global_min)
            cnt += 1
        if len(x) == 0 or x[len(x) - 1] < budget - 500:
            return
        while x[len(x) - 1] > budget:
            x.pop()
            y.pop()
        while x[len(x) - 1] < budget:
            x.append(len(x))
            y.append(y[len(y) - 1])
        assert len(x) == len(y)
        print(len(x))
        return np.array(x), np.array(y)


def build_boxplots_with_numbers():
    directory = 'best-designs-for-06-02-23'
    n, m = 5, 7
    fig, axs = plt.subplots(n, m)
    plt.subplots_adjust(wspace=1.15, hspace=0.50)
    design_ids = []
    designs_data = []
    for filename in os.listdir(directory):
        if filename.startswith('d'):
            full = os.path.join(directory, filename)
            selection = utils.read_selection(full)
            n = len(set(selection))
            desing_id = int(filename.split('d')[1])
            vals_file_name = f'vals_d{desing_id}'
            vals_full = os.path.join(directory, vals_file_name)
            vals = read_vals(vals_full)
            design_ids.append(desing_id)
            designs_data.append((n, vals))
    design_ids = np.array(design_ids)
    sorted_indexes = design_ids.argsort()
    cnt = 0
    for i in sorted_indexes:
        design_id = design_ids[i]
        n, vals = designs_data[i]
        row = cnt // m
        col = cnt % m
        axs[row, col].boxplot(x=vals, vert=True, showfliers=False)
        axs[row, col].set_title(f'{n}')
        axs[row, col].grid()
        axs[row, col].set_xticks([])
        cnt += 1
    fig.savefig('all-box-plots.pdf')
    plt.close()


def process_data_folder(data_folder, nruns, budget):
    good_x = []
    ys = []
    for i in range(nruns):
        path = data_folder + f'/everyeval-{i}/data_f25_SRON_nCH4_noisy_recovery/IOHprofiler_f25_DIM640.dat'
        print(path)
        res = extract_bestsofar(path, budget)
        if res:
            good_x, y = res
            ys.append(y)

    Y = np.array(ys)
    y1 = []
    err = []
    for i in range(len(Y[0])):
        y1.append(np.nanmean(Y[:, i]))
        err.append(np.nanstd(Y[:, i]))

    y1 = np.array(y1)
    err = np.array(err)
    return good_x, y1, err


def build_averaged_convergence(data_folder, nruns=100, budget=12000):
    nruns = int(nruns)
    budget = int(budget)
    good_x = []
    ys = []
    for i in range(nruns):
        path = data_folder + f'/everyeval-{i}/data_f25_SRON_nCH4_noisy_recovery/IOHprofiler_f25_DIM640.dat'
        print(path)
        res = extract_bestsofar(path, budget)
        if res:
            good_x, y = res
            ys.append(y)

    Y = np.array(ys)
    y1 = []
    err = []
    for i in range(len(Y[0])):
        y1.append(np.nanmean(Y[:, i]))
        err.append(np.nanstd(Y[:, i]))

    y1 = np.array(y1)
    err = np.array(err)
    print(len(ys))
    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    ax = plt.gca()
    ax.set_yscale('log')

    li, = plt.plot(good_x, y1, c='red')
    ax.fill_between(good_x, y1 - err, y1 + err, facecolor='red', alpha=0.20)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.yticks([1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.3, 2.6, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], fontsize=8)

    plt.legend([li], ['$(15 + 30)$ PhEA with G2 based mutation'])
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, '$100 \cdot$ std($S(x)$)/nCH4', va='center', rotation='vertical')
    fig.savefig('averaged-convergence-g2.pdf')
    plt.close()


def build_averaged_convergence1(nruns=100, budget=12000, cnt=2):
    data_folders = ['results-16-02-23/new-code-gen-ea/', 'results-16-02-23/new-code-gen-2/']
    nruns = int(nruns)
    budget = int(budget)
    cnt = int(cnt)

    x = []
    ys = []
    errs = []
    for data_folder in data_folders:
        x, y, err = process_data_folder(data_folder, nruns, budget)
        ys.append(y)
        errs.append(err)

    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    ax = plt.gca()
    ax.set_yscale('log')
    colors = ['red', 'blue', 'green']
    lis = []
    for i in range(cnt):
        y, err, cl = ys[i], errs[i], colors[i % len(colors)]
        li, = plt.plot(x, y, c=cl)
        lis.append(li)
        ax.fill_between(x, y - err, y + err, facecolor=cl, alpha=0.20)
    rvs = read_vals('sron_guess_precision')
    sron_guess_value = np.mean(rvs)
    sron_guess_err = np.std(rvs)
    li_guess, = plt.plot(x, [sron_guess_value] * len(ys[0]), c='black', linestyle='--')
    lis.append(li_guess)
    ax.fill_between(x, [sron_guess_value - sron_guess_err] * len(ys[0]),
                    [sron_guess_value + sron_guess_err] * len(ys[0]), facecolor='black', alpha=0.20)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    yticks = [1.6, 1.7, 1.9, 2.1, 2.4, 2.6, 2.9, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(yticks))
    plt.yticks(yticks, fontsize=8)

    plt.legend(lis,
               ['$(15 + 30)$ PhEA with Uniform EA based mutation', '$(15 + 30)$ PhEA with G2 based mutation',
                'SRON Guess'])
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, '$100 \cdot$ std($S(x)$)/nCH4', va='center', rotation='vertical')
    fig.savefig('averaged-convergence-g2.pdf')
    plt.close()

    return 'Success'


def extract_design(data_folder, budget=12000):
    budget = int(budget)
    if not extract_bestsofar(data_folder + '/data_f25_SRON_nCH4_noisy_recovery/IOHprofiler_f25_DIM640.dat', budget):
        print('Design is not successful')
        return
    path = data_folder + '/IOHprofiler_f25_SRON_nCH4_noisy_recovery.json'
    with open(path, 'r') as f:
        jobj = json.load(f)
        individual = jobj['scenarios'][0]['runs'][0]['best']['x']
    if not individual:
        print('Design not found')
        return
    a = ' '.join(str(i) for i in individual)
    return a


def precision_statistics(design_path, out):
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    measure_noise.MeasurePrecision(instrument, constants, design_path, out).process()


def extract_all_designs(experiment_folder, output_folder, nruns=100, budget=12000):
    nruns, budget = int(nruns), int(budget)
    a = []
    for experiment_id in range(0, 100):
        path = experiment_folder + f'/everyeval-{experiment_id}'
        print(path)
        d = extract_design(path, budget=budget)
        if not d:
            continue
        with open(output_folder + f'/d{experiment_id}', 'w') as f:
            print(d, file=f)
            a.append(experiment_id)
    print(f'Number of successful is {len(a)}')
    return a


def compare_distances_in_designs(designs_folder):
    selections = []
    for filename in os.listdir(designs_folder):
        d_name = os.path.join(designs_folder, filename)
        selections.append(utils.read_selection(d_name))
    n = len(selections)
    dists = np.zeros((n, n), dtype=float)

    inst = utils.create_instrument()
    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(d0).create_sequence_distance('kirill')

    for i in range(n):
        for j in range(i, n):
            dists[i][j] = d1(selections[i], selections[j])
            dists[j][i] = dists[i][j]
        print(*dists[i], flush=True)


if __name__ == '__main__':
    print(globals()[sys.argv[1]](*sys.argv[2:]))
