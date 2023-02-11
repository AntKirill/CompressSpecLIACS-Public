import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
import scipy
import sklearn.manifold
from matplotlib import pyplot as plt

import instrumentsimulator
import mylogger


def read_selection(file_name):
    with open(file_name, 'r') as f:
        return list(map(int, f.read().split()))


def create_instrument():
    instrument_settings = instrumentsimulator.InstrumentSettings()
    instrument = instrumentsimulator.InstrumentSimulator(instrument_settings)
    return instrument


@dataclass
class SRONConstants:
    nCH4: int
    albedo: float
    sza: int


class SearchSpaceDimReduction:
    def __init__(self, original_dim: int, reduced_dim: int) -> None:
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim

    def to_reduced(self, original):
        return original

    def to_original(self, reduced):
        return reduced


class SegmentsDimReduction(SearchSpaceDimReduction):
    def __init__(self, original_dim: int, reduced_dim: int) -> None:
        super().__init__(original_dim, reduced_dim)
        self.m = self.original_dim // self.reduced_dim
        self.r = self.original_dim % self.reduced_dim

    def to_reduced(self, original):
        reduced = np.zeros(self.reduced_dim, dtype=int)
        cnt = 0
        for i in range(self.r):
            reduced[cnt] = (original[i * (self.m + 1)])
            cnt += 1
        for i in range(self.r, self.reduced_dim):
            reduced[cnt] = original[i * self.m]
            cnt += 1
        assert cnt == self.reduced_dim, f'cnt = {cnt}, nsegments = {self.reduced_dim}'
        return reduced

    def to_original(self, reduced):
        original = np.zeros(self.original_dim, dtype=int)
        cnt = 0
        for i in range(self.r):
            for j in range(self.m + 1):
                original[cnt] = reduced[i]
                cnt += 1
        for i in range(self.r, self.reduced_dim):
            for j in range(self.m):
                original[cnt] = reduced[i]
                cnt += 1
        assert cnt == self.original_dim, f'cnt = {cnt}, M = {self.original_dim}'
        return original


class SequenceFiltersVisualization:
    def __init__(self, instrument, constants: SRONConstants) -> None:
        self.instrument = instrument
        self.constants = constants

    def _get_transmission_profiles(self, sequence):
        selectedfilters = self.instrument.getTransmissionMatrix(sequence)
        return [selectedfilters.T[:, i] for i in range(len(sequence))]

    def _get_spectral_range(self, sequence):
        f = ObjFunctionPure(self.instrument, self.constants)
        f(sequence)  # Need to call in order to get spectral_range for the instrument
        return self.instrument.spectral_range

    def save_transmission_profiles(self, sequence, file_name, p: tuple):
        spectral_range = self._get_spectral_range(sequence)
        profiles = self._get_transmission_profiles(sequence)
        n, m = p
        sz = len(profiles)
        fig, axs = plt.subplots(n, m)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        def get_axs(i, j):
            if n == 1 or m == 1:
                return axs[i + j]
            return axs[i, j]

        for i in range(n):
            for j in range(m):
                profile_number = i * m + j
                if profile_number < sz:
                    get_axs(i, j).plot(spectral_range, profiles[profile_number], linewidth=0.4)
                get_axs(i, j).set_xticks([])
                get_axs(i, j).set_yticks([])
                [i.set_linewidth(0.1) for i in get_axs(i, j).spines.values()]
                # get_axs(i, j).axis('off')

        colors = ['red', 'blue', 'orange', 'purple', 'cyan', 'green']
        cnt = 0
        is_first = True
        cur_color = colors[0]
        for i in range(n):
            for j in range(m):
                profile_number = i * m + j
                if profile_number < sz:
                    if profile_number < sz - 1 and sequence[profile_number + 1] == sequence[profile_number]:
                        if is_first:
                            cur_color = colors[cnt]
                            cnt = (cnt + 1) % len(colors)
                            is_first = False
                        [i.set_linewidth(2) for i in get_axs(i, j).spines.values()]
                        [i.set_edgecolor(cur_color) for i in get_axs(i, j).spines.values()]
                    elif not is_first:
                        [i.set_linewidth(2) for i in get_axs(i, j).spines.values()]
                        [i.set_edgecolor(cur_color) for i in get_axs(i, j).spines.values()]
                        is_first = True

        fig.text(0.5, 0.07, 'wavelength (nm)', ha='center')
        fig.text(0.09, 0.5, 'transmission', va='center', rotation='vertical')
        fig.savefig(file_name)
        plt.close()


class LandscapeVisualization:
    def __init__(self) -> None:
        pass

    def visualize(self, X, f, d1):
        tsne = sklearn.manifold.TSNE(perplexity=50, metric=d1)
        y = tsne.fit_transform(X)
        fig = plt.figure()
        jet_cmap = mpl.colormaps['jet']
        # values = np.random.uniform(0.0004, 0.002, len(X))
        values = np.array([f(x) for x in X])
        mi, ma = values.min(), values.max()
        values = np.array([(v - mi) / (ma - mi) for v in values])
        plt.scatter(y[:len(y) - 1, 0], y[:len(y) - 1, 1], c=[jet_cmap(i) for i in values[:len(y) - 1]])
        plt.scatter(y[len(y) - 1, 0], y[len(y) - 1, 1], c=[jet_cmap(values[len(y) - 1])], marker='x')
        fig.savefig('manifold.pdf')
        plt.close()


class SegmentedSequenceFiltersVisualization(SequenceFiltersVisualization):
    def __init__(self, instrument, constants, segm_dim_reducer: SegmentsDimReduction) -> None:
        super().__init__(instrument, constants)
        self.segm_dim_reducer = segm_dim_reducer

    def _get_transmission_profiles(self, reduced_sequence):
        sequence_original_space = self.segm_dim_reducer.to_original(reduced_sequence)
        original_profiles = super()._get_transmission_profiles(sequence_original_space)
        reduced_profiles = np.zeros(self.segm_dim_reducer.reduced_dim, dtype=np.ndarray)
        cnt = 0
        for i in range(self.segm_dim_reducer.r):
            reduced_profiles[cnt] = (original_profiles[i * (self.segm_dim_reducer.m + 1)])
            cnt += 1
        for i in range(self.segm_dim_reducer.r, self.segm_dim_reducer.reduced_dim):
            reduced_profiles[cnt] = original_profiles[i * self.segm_dim_reducer.m]
            cnt += 1
        return reduced_profiles

    def _get_spectral_range(self, reduced_sequence):
        sequence_original_space = self.segm_dim_reducer.to_original(reduced_sequence)
        return super()._get_spectral_range(sequence_original_space)


class AbstractObjectiveFunctionSRON(ABC):
    def __init__(self, instrument, constants: SRONConstants):
        self.instrument = instrument
        self.sron_constants = constants

    @abstractmethod
    def __call__(self, selection):
        pass


class ObjFunctionPure(AbstractObjectiveFunctionSRON):
    def __call__(self, selection):
        _, bias, = self.instrument.simulateMeasurement(selection,
                                                       nCH4=self.sron_constants.nCH4,
                                                       albedo=self.sron_constants.albedo,
                                                       sza=self.sron_constants.sza,
                                                       n=1,
                                                       verbose=False)
        return bias


class ObjFunctionAverageSquare(ObjFunctionPure):
    def __init__(self, instrument, constants, noisy_evals_number=1000):
        super().__init__(instrument, constants)
        self.N = noisy_evals_number
        self.obj_values = np.zeros(self.N)

    def __call__(self, selection):
        for i in range(self.N):
            self.obj_values[i] = super().__call__(selection)
        return np.nanmean(self.obj_values ** 2)

    @property
    def sron_bias(self):
        return 100 * np.nanmean(self.obj_values)

    @property
    def sron_precision(self):
        return 100 * np.std(self.obj_values)


class FilterDistanceFactory:
    def __init__(self, instrument) -> None:
        self.instrument = instrument

    def create_filters_distance(self, method):
        return lambda filter1, filter2: self.instrument.getDistance(filter1, filter2, method)

    def create_precomputed_filters_distance(self, method, file_name=None):
        dist = self.create_precomputed_filter_distance_matrix(method, file_name)
        return lambda filter1, filter2: dist[filter1, filter2]

    def create_precomputed_filter_distance_matrix(self, method, file_name=None):
        lib_size = self.instrument.filterlibrarysize
        dist = np.zeros((lib_size, lib_size))
        if file_name is None:
            for i in range(lib_size):
                for j in range(lib_size):
                    dist[i, j] = self.instrument.getDistance(i, j, method)
        else:
            assert file_name.find(f'method{method}') != -1
            with open(file_name, 'r') as f:
                for i in range(lib_size):
                    for j in range(lib_size):
                        i_, j_, d = f.readline().split()
                        assert int(i_) == i and int(j_) == j
                        dist[i, j] = float(d)
        return dist


class SequenceDistanceKirill:
    def __init__(self, d0) -> None:
        self.d0 = d0

    def __call__(self, seq1, seq2):
        return self.compute_all(seq1, seq2)[2]

    def get_permutation(self, seq1, seq2):
        return self.get_permutation_and_value(seq1, seq2)[0]

    def get_permutation_and_value(self, seq1, seq2):
        ind1, ind2, value = self.compute_all(seq1, seq2)
        permutation = np.zeros(len(seq1), dtype=int)
        for i in range(len(seq1)):
            permutation[ind1[i]] = ind2[i]
        return permutation, value

    def compute_all(self, s1, s2):
        d = np.zeros((len(s1), len(s2)))
        for i in range(len(s1)):
            for j in range(len(s2)):
                d[i, j] = self.d0(s1[i], s2[j])
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(d)
        return row_ind, col_ind, d[row_ind, col_ind].sum()


class SequenceDistanceFactory:
    def __init__(self, d0) -> None:
        self.d0 = d0

    def create_sequence_distance(self, method_sequence_distance):
        if method_sequence_distance == 'kirill':
            seq_dist = SequenceDistanceKirill(self.d0)
            return seq_dist


def add_logger(f, ndim: int, root_name: str, alg_name: str, alg_info: str, generator=None):
    wrapped_f = mylogger.MyObjectiveFunctionWrapper(f, dim=ndim, fname='SRON_nCH4_noisy_recovery')
    global logger
    logger = mylogger.MyLogger(root=root_name,
                               folder_name="everyeval",
                               algorithm_name=alg_name,
                               algorithm_info=alg_info)
    logger_best = mylogger.MyLogger(root=root_name,
                                    folder_name="bestsofar",
                                    algorithm_name=alg_name,
                                    algorithm_info=alg_info,
                                    logStrategy=mylogger.LoggingBestSoFar,
                                    isLogArg=True)
    logger.watch(f, ['sron_bias', 'sron_precision'])
    logger_best.watch(f, ['sron_bias', 'sron_precision'])
    if generator is not None:
        logger.watch(generator, ['distance_from_parent', 'target_distance_from_parent'])
    wrapped_f.attach_logger(logger)
    wrapped_f.attach_logger(logger_best)
    return wrapped_f


def add_segm_dim_reduction(f, original_dim, reduced_dim):
    dim_reduction = SegmentsDimReduction(original_dim, reduced_dim)
    return lambda reduced_arg: f(dim_reduction.to_original(reduced_arg))


if __name__ == '__main__':
    print(globals()[sys.argv[1]](*sys.argv[2:]))
