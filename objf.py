import instrumentsimulator
import numpy as np


class ObjFunSRON:
    def __init__(self, rep):
        instrument_settings = instrumentsimulator.InstrumentSettings()
        self.instrument = instrumentsimulator.InstrumentSimulator(instrument_settings)
        self.rep = rep
        self.search_space_dim = 640

    def __call__(self, original):
        self.x = np.zeros(self.rep)
        for i in range(self.rep):
            _, self.x[i], = self.instrument.simulateMeasurement(original, nCH4=2000, albedo=0.15, sza=70, n=1,
                                                                verbose=False)
        return np.mean(self.x ** 2)

    @property
    def sron_bias(self):
        return 100 * np.nanmean(self.x)

    @property
    def sron_precision(self):
        return 100 * np.std(self.x)


class ProfiledObjFunSRON:
    def __init__(self, of, config):
        import utils
        self.of = of
        self.obj_f_wrapped = utils.add_logger(self.of, of.search_space_dim, config.folder_name, config.algorithm, config.algorithm_info)

    def __call__(self, x):
        return self.obj_f_wrapped(x)

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

    def __call__(self, x):
        y = self.dim_red.to_original(x)
        return self.of(y)

    @property
    def search_space_dim(self):
        return self.dim_red.reduced_dim

    @property
    def instrument(self):
        return self.of.instrument


def create_profiled_obj_fun_for_reduced_space(config):
    return ReducedDimObjFunSRON(config.n_segms, ProfiledObjFunSRON(ObjFunSRON(config.n_reps), config))

