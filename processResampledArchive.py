import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# experiment_root = 'archives/archiveResampling'

def main(experiment_root):
    design_ids = []
    designs_data = []
    nums = []
    for item in os.listdir(experiment_root):
        fullpath = os.path.join(experiment_root, item)
        if os.path.isfile(fullpath) and item.startswith('samples_for_'):
            id = item.split('_')[4]
            with open(fullpath, 'r') as file:
                for row in file:
                    sp = row.split(' ')
                    selection = [int(k) for k in sp[0:640]]
                    samples = np.array([float(k) for k in sp[640:]])**2
                    n = len(set(selection))
                    design_ids.append(id)
                    designs_data.append([id, n, samples])
                    nums.append(n)

    fig = plt.figure()
    ax = plt.gca()
    design_ids = np.array(design_ids)
    designs_data.sort(key=lambda d: np.mean(d[2]))
    x = []
    ticks = []
    colors = []
    colors_lib = mpl.cm.jet(np.linspace(0, 1, len(set(nums))))
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
    bplot = ax.boxplot(x, vert=True, showfliers=False, patch_artist=True, usermedians=[np.mean(d[2]) for d in designs_data])
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
    # fig.text(0.04, 0.5, 'Performance in 1K runs', va='center', rotation='vertical')
    ax.grid()
    fig.savefig('box-plots.pdf')
            

if __name__ == '__main__':
    main(*sys.argv[1:])