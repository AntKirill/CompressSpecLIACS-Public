# Begin methods
# %%

import sys
import instrumentsimulator
import numpy as np
import sys
import objf
from scipy import stats
import utils
import algorithms as algs
import umda
from myrandom import RandomEngine
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


class CriteriaD:
    def __init__(self):
        instrument_settings = instrumentsimulator.InstrumentSettings()
        self.instrument = instrumentsimulator.InstrumentSimulator(instrument_settings)
        self.search_space_dim = 640
        self.x = None

    def __call__(self, original):
        _, value, = self.instrument.simulateMeasurement(original, nCH4=2000, albedo=0.15, sza=70, n=1, verbose=False)
        return value


class CriteriaF:
    def __init__(self, D):
        self.D = D
        self.instrument = D.instrument
        self.values = None
        self.search_space_dim = 640

    def __call__(self, x, reps):
        self.values = np.zeros(reps)
        for i in range(reps):
            self.values[i] = self.D(x)
        return np.mean(self.values**2)

    def get_measurements(self):
        return self.values


class ProfiledF:
    def __init__(self, of, config):
        import utils
        self.of = of
        self.obj_f_wrapped = utils.add_logger(self.of, of.search_space_dim, config.folder_name, config.algorithm, config.algorithm_info)

    def __call__(self, x, reps):
        return self.obj_f_wrapped(x, reps)

    def get_measurements(self):
        return self.of.get_measurements()

    @property
    def instrument(self):
        return self.of.instrument

    @property
    def search_space_dim(self):
        return self.of.search_space_dim

class ReducedDimObjFunSRON:
    def __init__(self, l: int, of):
        import utils
        self.of = of
        self.dim_red = utils.SegmentsDimReduction(of.search_space_dim, l)

    def __call__(self, x, reps):
        y = self.dim_red.to_original(x)
        return self.of(y, reps)

    def get_measurements(self):
        return self.of.get_measurements()

    @property
    def search_space_dim(self):
        return self.dim_red.reduced_dim

    @property
    def instrument(self):
        return self.of.instrument


def create_profiled_obj_fun_for_reduced_space(config):
    return ReducedDimObjFunSRON(config.n_segms, ProfiledF(CriteriaF(CriteriaD()), config))


def sort_dist_matrix(matrix):
    matrix_sorted_rows = []
    for i in range(len(matrix)):
        tmp = [(matrix[i][j], j) for j in range(len(matrix[i]))]
        tmp.sort()
        matrix_sorted_rows.append(tmp)
    return matrix_sorted_rows


class DxSamplesContainer:
    def __init__(self, F, x):
        self.f = F
        self.x = x
        self.D_samples = []

    def add_samples(self, n):
        self.f(self.x, n)
        st = self.f.get_measurements()
        self.D_samples = np.concatenate([self.D_samples, st])
        return self.D_samples


def on_reload():
    global D, F, L, Fr, x0, dist_matrix, dist_matrix_sorted, dist
    D = CriteriaD()
    F = CriteriaF(D)
    L = D.instrument.filterlibrarysize
    Fr = ReducedDimObjFunSRON(16, F)
    x0 = D.instrument.filterguess()
    dist_matrix = utils.create_dist_matrix(F, 2)
    dist_matrix_sorted = sort_dist_matrix(dist_matrix)
    dist = utils.create_distance(F, dist_matrix, 'kirill')


def plot_dist():
    x = utils.generate_random_solution(16, L)
    xSamples = DxSamplesContainer(Fr, x)
    D_samples = xSamples.add_samples(1000)
    print(np.mean(D_samples)**2/np.var(D_samples))

    fig, ax = plt.subplots()
    # lambda_ = 1 / np.mean(st)
    # dx = (max(st) - min(st)) / 100
    # x_ = np.arange(min(st), max(st), dx)
    # y = [4000 * (np.exp(-lambda_ * x) - np.exp(-lambda_ * (x + dx))) for x in x_]
    plt.hist(D_samples, bins=50)
    # plt.plot(x_, y)
    plt.show()


def experiment_ratio_mean_var(N):
    means = []
    variances = []
    for i in range(N):
        x = utils.generate_random_solution(16, L)
        container = DxSamplesContainer(Fr, x)
        obj_value = Fr(x, 1000)
        Dx = Fr.get_measurements()
        m = np.mean(Dx)
        v = np.var(Dx)
        means.append(m)
        variances.append(v)
        print(obj_value, m**2 + v, m**2/v)
    return np.array(means), np.array(variances)


def build_surface_mean_var():
    F = lambda m, s: (s-1)/(1-m)

    MMIN, MMAX, MSTEP = 0.0, 0.998, 0.001
    SMIN, SMAX, SSTEP = 1.000001, 1.005, 0.000001
    KMIN, KMAX, KSTEP = 0.0, 8, 0.5

    X = np.arange(MMIN, MMAX, MSTEP)
    Y = np.arange(SMIN, SMAX, SSTEP)
    x_ticks = np.arange(MMIN, MMAX, 0.15)
    y_ticks = np.arange(1, SMAX, 0.0005)
    # z_ticks = np.arange(KMIN, KMAX, KSTEP)
    X, Y = np.meshgrid(X, Y)
    Z = X.copy()
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            if F(X[i][j], Y[i][j]) < 0.0000001:
                breakpoint()
            Z[i][j] = math.log(F(X[i][j], Y[i][j]))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10, 8))
    # ax.set_xlabel(r'$m = \left(\dfrac{E(x)}{E(y)}\right)^2$')
    # ax.set_ylabel(r'$s = \left(\dfrac{{Var}(x)}{{Var}(y)}\right)^2$')
    # ax.set_zlabel('$\dfrac{s-1}{1-m}$', rotation=0)
    # ax.set_title('')

    mycmap = mpl.cm.jet

    # ax.plot_surface(X, Y, Z, alpha=0.7, cmap=mycmap, antialiased=True, linewidth=0.2, edgecolor='k')
    special_z = -6.64
    Z1 = np.full_like(Z, special_z)
    ax.plot_surface(X, Y, np.where(Z <= special_z, Z, np.nan), alpha=0.7, color=mycmap(0), linewidth=0.4, edgecolor='k')
    ax.plot_surface(X, Y, Z1, alpha=0.8, color='gray', linewidth=0.2, edgecolor='k')
    ax.plot_surface(X, Y, np.where(Z >= special_z, Z, np.nan), alpha=0.7, cmap=mycmap, linewidth=0.4, edgecolor='k')

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    # ax.set_zticks(z_ticks)

    # sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(Z.min(), Z.max()))
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax, label='Colorbar Label')

    # ax.contour(X, Y, Z, zdir='z', offset=Z.min()-1, cmap='coolwarm', extend='both')
    # ax.contourf(X, Y, Z, zdir='z', offset=Z.min()-1, cmap='coolwarm', extend='both')
    ax.view_init(30, 168)
    plt.show()

# End methods
# %%

on_reload()
means, variances = experiment_ratio_mean_var(1000)
with open('mean_var_test.csv', 'w') as file:
    print('mean, var', file=file)
    for m, v in zip(means, variences):
        print(m, v, file=file)
with open('ans.txt', 'w') as file:
    print(np.log(np.mean(means**2/variances)), file=file)

