import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import utils
import measure_noise


def read_vals(file_name):
    with open(file_name, 'r') as f:
        return list(map(float, f.read().split()))


def build_box_plot(values_file, out, suffix=''):
    vals = read_vals(values_file)
    fig = plt.figure()
    plt.boxplot(x=vals, vert=False)
    plt.grid()
    if len(suffix) > 0:
        fig.savefig(f'{out}/box-plot_{suffix}.pdf')
    else:
        fig.savefig(f'{out}/box-plot.pdf')
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


def build_tp(design_file, out, R=16, n=4, m=4, suffix='', instrument=None):
    if not instrument:
        instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    selection = utils.read_selection(design_file)
    M = 640
    R, n, m = int(R), int(n), int(m)
    if R < M:
        dim_reduction = utils.SegmentsDimReduction(M, R)
        selection = dim_reduction.to_reduced(selection)
        vis = utils.SegmentedSequenceFiltersVisualization(
            instrument, constants, dim_reduction)
    else:
        vis = utils.SequenceFiltersVisualization(instrument, constants)
    selection.sort()

    vis.save_transmission_profiles(selection, f'{out}/tp_{suffix}.pdf', (n, m))
    sz = len(set(selection))
    print(f'Number of different filters is {sz}')


def build_pdf(values_file, out, suffix=''):
    vals = read_vals(values_file)
    fig = plt.figure()
    plt.hist(vals, bins=5)
    fig.savefig(f'{out}/pdf_{suffix}.pdf')
    plt.close()


def process_designs(designs_folder):
    instrument = utils.create_instrument()
    for filename in os.listdir(designs_folder):
        if filename.startswith('d'):
            d_filename = f'{designs_folder}/{filename}'
            vals_filename = f'{designs_folder}/vals_{filename}'
            if not os.path.exists(vals_filename):
                continue
            build_box_plot(vals_filename, designs_folder, filename)
            build_pdf(vals_filename, designs_folder, filename)
            selection = utils.read_selection(d_filename)
            ndiff = len(set(selection))
            if ndiff <= 4:
                R, n, m = 4, 2, 2
            elif ndiff <= 16:
                R, n, m = 16, 4, 4
            elif ndiff <= 32:
                R, n, m = 32, 8, 4
            print(f'Design {filename}')
            build_tp(d_filename, designs_folder, R, n, m, filename, instrument)


def generate_tex():
    with open('template_tex.tex', 'r') as f:
        template_str = f.read()
    # ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32,
        # 33, 36, 37, 42, 47]
    ids = [i for i in range(35)]
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
            data.append(MyEval(int(row[0]), float(row[1]), float(
                row[2]), float(row[3]), [int(i) for i in row[4:]]))

    d0 = None
    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(
        2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(
        d0=d0, instrument=inst, M=640, R=640).create_sequence_distance('kirill')
    dists = np.zeros(len(data) - 1)
    for i in range(1, len(data)):
        dists[i-1] = d1(data[i-1].design, data[i].design)
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
        if len(x) == 0 or x[len(x) - 1] < 0.1*budget:
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


def extract_good_designs(budget):
    dirs = ['results-16-02-23', 'results-20-02-23', 'results-22-02-23']
    outpath = 'good-designs-22-02-23'
    cnt = 0
    for directory in dirs:
        for x in os.walk(directory):
            mypath = x[0]
            mydir = os.path.basename(mypath)
            if mydir.startswith('everyeval'):
                d = extract_design(mypath, budget)
                with open(f'{outpath}/d{cnt}', 'w') as f:
                    print(d, file=f, flush=True)
                cnt += 1
    print(cnt)
    return 'Success'


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


def build_boxplots_with_numbers1():
    # directory = 'best-designs-for-06-02-23'
    directory = 'good-designs-22-02-23'
    # dir1 = directory + '/new-code/'
    dir1 = directory
    pdf_name = 'all-box-plots-new-code-r.pdf'
    # n, m = 1, 35
    # fig, axs = plt.subplots(n, m)
    # plt.subplots_adjust(wspace=1.15, hspace=0.50)
    design_ids = []
    designs_data = []
    nums = []
    for filename in os.listdir(directory):
        if filename.startswith('d'):
            full = os.path.join(directory, filename)
            selection = utils.read_selection(full)
            n = len(set(selection))
            desing_id = int(filename.split('d')[1])
            vals_file_name = f'vals_d{desing_id}'
            vals_full = os.path.join(directory, vals_file_name)
            vals = read_vals(vals_full)
            vals1 = read_vals(os.path.join(dir1, vals_file_name))
            old_mean = np.mean(vals)
            new_mean = np.mean(vals1)
            bias = new_mean - old_mean
            for i in range(len(vals)):
                vals[i] += bias
            eps = 0.0000001
            assert new_mean - eps < np.mean(vals) < new_mean + eps
            design_ids.append(desing_id)
            designs_data.append([desing_id, n, vals])
            nums.append(n)
    fig = plt.figure()
    ax = plt.gca()
    design_ids = np.array(design_ids)
    designs_data.sort(key=lambda d: np.mean(d[2]))
    x = []
    ticks = []
    colors = []
    colors_lib = mpl.cm.turbo(np.linspace(0, 1, len(set(nums))))
    unique_nums = list(set(nums))
    unique_nums.sort()
    n_to_color = {}
    for n, c in zip(unique_nums, colors_lib):
        n_to_color[n] = c
    # n_to_color = {11: 'green', 12: 'blue', 13: 'purple',
        # 14: 'orange', 15: 'red', 16: 'black'}
    color_to_bplot_number = {}
    num_to_bplot_number = {}
    cnt = 0
    for d in designs_data:
        ticks.append(d[0])
        x.append(d[2])
        colors.append(n_to_color[d[1]])
        # color_to_bplot_number[n_to_color[d[1]]] = cnt
        num_to_bplot_number[d[1]] = cnt
        cnt += 1
    bplot = ax.boxplot(x, vert=True, showfliers=False, patch_artist=True)
    cnt = 0
    for color in colors:
        bplot['boxes'][cnt].set_facecolor(color)
        cnt += 1
    bplot_boxes = []
    legends = []
    for i in n_to_color.keys():
        num = num_to_bplot_number[i]
        n_diff_filters = designs_data[num][1]
        bplot_boxes.append(bplot['boxes'][num])
        legends.append(f'{n_diff_filters} diff filters')
    plt.xticks([i for i in range(1, len(x) + 1)], ticks, fontsize=6)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bplot_boxes, legends, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.text(0.5, 0.04, 'Design ID', ha='center')
    fig.text(0.04, 0.5, 'Performance in 1K runs',
             va='center', rotation='vertical')
    ax.grid()
    fig.savefig(pdf_name)
    plt.close()
    return 'Success'


def process_data_folder(data_folder, nruns, budget):
    good_x = []
    ys = []
    for i in range(nruns):
        path = data_folder + \
            f'/everyeval-{i}/data_f25_SRON_nCH4_noisy_recovery/IOHprofiler_f25_DIM640.dat'
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
        path = data_folder + \
            f'/everyeval-{i}/data_f25_SRON_nCH4_noisy_recovery/IOHprofiler_f25_DIM640.dat'
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
    plt.yticks([1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.3, 2.6, 3.0,
               4.0, 5.0, 6.0, 7.0, 8.0, 9.0], fontsize=8)

    plt.legend([li], ['$(15 + 30)$ PhEA with G2 based mutation'])
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, '$100 \cdot$ std($S(x)$)/nCH4',
             va='center', rotation='vertical')
    fig.savefig('averaged-convergence-g2.pdf')
    plt.close()


def build_averaged_convergence1(nruns=100, budget=12000):
    data_folders = ['results-16-02-23/new-code-gen-ea/',
                    'results-16-02-23/new-code-gen-2/']
    colors = ['red', 'blue']
    # names = ['$(15 + 30)$ PhEA, LAP dist, Uniform Mut, PermBased approx',
    # '$(15 + 30)$ PhEA, LAP dist, G2']
    names = ['New EA, LAP(2)', 'Old EA, LAP(2)']
    pdf_name = 'averaged-convergence-ea-vs-g2.pdf'

    nruns = int(nruns)
    budget = int(budget)
    cnt = len(data_folders)

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
    lis = []
    for i in range(cnt):
        y, err, cl = ys[i], errs[i], colors[i % len(colors)]
        li, = plt.plot(x, y, c=cl)
        lis.append(li)
        ax.fill_between(x, y - err, y + err, facecolor=cl, alpha=0.20)
    rvs = read_vals('sron_guess_precision')
    sron_guess_value = np.mean(rvs)
    sron_guess_err = np.std(rvs)
    li_guess, = plt.plot(x, [sron_guess_value] *
                         len(ys[0]), c='black', linestyle='--')
    lis.append(li_guess)
    names.append('SRON Guess')
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

    plt.legend(lis, names)
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, '$100 \cdot$ std($S(x)$)/nCH4',
             va='center', rotation='vertical')
    fig.savefig(pdf_name)
    plt.close()

    return 'Success'


def build_averaged_convergence2(nruns=100, budget=12000):
    data_folders = ['results-20-02-23/seqdist-my/',
                    'results-20-02-23/seqdist-2/',
                    'results-20-02-23/seqdist-3/',
                    'results-20-02-23/experiments-extreme']
    # data_folders = ['results-22-02-23/new-dims/mk_2_s4_20-02-2023_20h',
    # 'results-20-02-23/experiments-extreme',
    # 'results-22-02-23/new-dims/mk_2_s32_20-02-2023_20h',
    # 'results-22-02-23/new-dims/m2_s4_20-02-2023_20h',
    # 'results-20-02-23/seqdist-2',
    # 'results-22-02-23/new-dims/m2_s32_20-02-2023_20h',
    # 'results-22-02-23/new-dims/m3_s4_20-02-2023_20h',
    # 'results-20-02-23/seqdist-3',
    # 'results-22-02-23/new-dims/m3_s32_20-02-2023_20h']
    colors = mpl.cm.rainbow(np.linspace(0, 1, len(data_folders)))
    # colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    # names = ['$(15 + 30)$ PhEA, LAP dist, Uniform Mut, PermBased approx',
    # '$(15 + 30)$ PhEA, method 2 dist, Uniform Mut, Simple approx',
    # '$(15 + 30)$ PhEA, method 3 dist, Uniform Mut, Simple approx',
    # '$(30 + 30)$ PhEA, LAP dist, Uniform Mut, PermBased approx']
    names = ['EA, $d_1$ with LAP(2), $R=16$, 5 runs',
            'EA, $d_1$ with method 2, $R=16$, 5 runs',
            'EA, $d_1$ with method 3, $R=16$, 2 runs',
            'Modified EA, $d_1$ with LAP(2), $R=16$, 1 run']
    # names = ['Modified EA, $d_1$ with LAP(2), $R = 4$, 3 runs',
             # 'Modified EA, $d_1$ with LAP(2), $R = 16$, 1 run',
             # 'Modified EA, $d_1$ with LAP(2), $R = 32$, 3 runs',
             # 'Modified EA, $d_1$ with method 2, $R = 4$, 3 runs',
             # 'EA, $d_1$ with method 2, $R = 16$, 5 runs',
             # 'Modified EA, $d_1$ with method 2, $R = 32$, 3 runs',
             # 'Modified EA, $d_1$ with method 3, $R = 4$, 1 run',
             # 'EA, $d_1$ with method 3, $R = 16$, 2 runs ',
             # 'Modified EA, $d_1$ with method 3, $R = 32$, 3 runs']
    pdf_name = 'averaged-convergence-new-dists.pdf'

    nruns = int(nruns)
    budget = int(budget)
    cnt = len(data_folders)

    x = []
    ys = []
    errs = []
    for data_folder in data_folders:
        x, y, err = process_data_folder(data_folder, nruns, budget)
        x = x[100:]
        y = y[100:]
        err = err[100:]
        ys.append(y)
        errs.append(err)

    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    ax = plt.gca()
    ax.set_yscale('log')
    lis = []
    for i in range(cnt):
        y, err, cl = ys[i], errs[i], colors[i % len(colors)]
        li, = plt.plot(x, y, c=cl)
        lis.append(li)
        ax.fill_between(x, y - err, y + err, facecolor=cl, alpha=0.20)
    rvs = read_vals('sron_guess_precision')
    sron_guess_value = np.mean(rvs)
    sron_guess_err = np.std(rvs)
    li_guess, = plt.plot(x, [sron_guess_value] *
                         len(ys[0]), c='black', linestyle='--')
    lis.append(li_guess)
    names.append('SRON Guess')
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
    # ax.set_xticks(list(ax.get_xticks()) + [100])

    plt.legend(lis, names)
    plt.grid(linewidth=0.4)
    fig.text(0.5, 0.02, 'Number of $f$ evaluations', ha='center')
    fig.text(0.06, 0.52, '$100 \cdot$ std($S(x)$)/nCH4',
             va='center', rotation='vertical')
    fig.savefig(pdf_name)
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


def precision_statistics(dirpath, out):
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    for filename in os.listdir(dirpath):
        if filename.startswith('d'):
            print(f'Run for {filename} ...', end='', flush=True)
            full = os.path.join(dirpath, filename)
            measure_noise.MeasurePrecision(
                instrument, constants, full, out).process()
            print('Done', flush=True)
    return 'Success'


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
    d0 = utils.FilterDistanceFactory(inst).create_precomputed_filters_distance(
        2, 'precomputedFiltersDists/method2.txt')
    d1 = utils.SequenceDistanceFactory(d0).create_sequence_distance('kirill')

    for i in range(n):
        for j in range(i, n):
            dists[i][j] = d1(selections[i], selections[j])
            dists[j][i] = dists[i][j]
        print(*dists[i], flush=True)


if __name__ == '__main__':
    print(globals()[sys.argv[1]](*sys.argv[2:]))
