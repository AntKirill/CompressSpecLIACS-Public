import csv
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser('Best-so-far plot creator')
parser.add_argument('--csv-paths', nargs='+', required=True,
                    help='Path to csv files with best-so-far data')
args = parser.parse_args()

ys = []
for path in args.csv_paths:
    with open(path, 'r') as f:
        r = csv.reader(f, delimiter=' ')
        next(r, None)
        global_min = float("inf")
        x = []
        y = []
        cnt = 0
        for row in r:
            v = float(row[3])
            if v < global_min:
                x.append(cnt)
                y.append(v)
                global_min = v
            cnt += 1
        if len(y) == 0:
            breakpoint()
        y.append(y[len(y)-1])
        x.append(1000)
        ys.append((x, y))

fig = plt.figure()
colors = ['blue', 'red', 'green', 'orange']
cnt = 0
l = []
for y in ys:
    li, = plt.plot(y[0], y[1], colors[cnt])
    l.append(li)
    cnt += 1

plt.legend(l, ['NoSegmRandom', 'SegmRandom', 'NoSegmInit', 'SegmInit'])
plt.grid()
fig.savefig('chart.pdf')
