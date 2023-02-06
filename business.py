import csv
import json
import os
import sys

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

    plt.legend([li], ['EA-16Pheno-Random'])
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, '$100 \cdot$ std($S(x)$)/nCH4', va='center', rotation='vertical')
    fig.savefig('averaged-convergence.pdf')
    plt.close()


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
