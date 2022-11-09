# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:58:57 2022

@author: marijns
"""
import numpy as np
from dataclasses import dataclass
from RadianceModel import RadianceModel
import os
import csv
from scipy import optimize
import pandas as pd
import time
module_directory = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Detector:
    pxlsize: float = 15e-6
    npxl_alt: int = 640
    npxl_act: int = 512
    readnoise: float = 50
    darkcurrent: float = 2000
    quantumeff: float = 0.85
    fullwellcapicity: float = 450e4


@dataclass
class InstrumentSettings:
    resolution: float = 150
    oversampling: float = 2.5
    aperture_act: float = 30e-3
    aperture_alt: float = 30e-3
    sza: float = 70
    vza: float = 0
    detector = Detector()


class InstrumentSimulator():

    earthradius = 6.371e6       # in m
    altitude = 500e3
    orbital_speed = np.sqrt(3.986e14 / (altitude + earthradius))
    projected_orbital_speed = orbital_speed * earthradius / \
        (altitude + earthradius)    # orbital speed projected on earth surface

    wavelength_min = 1625
    wavelength_max = 1670
    wavelength_n = 225

    def __init__(self, instrumentsettings):

        self.instrumentsettings = instrumentsettings
        # Ground-projected instantaneous field-of-view
        self.giFOV = self.instrumentsettings.resolution / \
            self.instrumentsettings.oversampling
        # swath ACT in m
        self.swath_act = self.giFOV * self.instrumentsettings.detector.npxl_act
        # swath ALT in m
        self.swath_alt = self.giFOV * self.instrumentsettings.detector.npxl_alt

        self.focallength = self.altitude / self.giFOV * \
            self.instrumentsettings.detector.pxlsize             # in m
        self.fnumber_alt = self.focallength / self.instrumentsettings.aperture_alt
        self.fnumber_act = self.focallength / self.instrumentsettings.aperture_act
        self.etendue = self.instrumentsettings.aperture_alt * \
            self.instrumentsettings.aperture_act * \
            self.giFOV**2 / self.altitude**2  # in sr m2

        self.integrationtime = self.giFOV / \
            self.projected_orbital_speed                              # in seconds
        # in Hz
        self.readoutfreq = 1 / self.integrationtime

        self.loadFilterLibrary()
        self.loadRadianceModel()

    def loadOptimizedFilterset(self):
        filtersPath = os.path.join(
            module_directory, "FilterLibrary/optimized_set_200_filters_225.txt")
        data_string = [
            "Transmission", 'lc!geomparam!thickness!shape!Pulse_time!x$y$z!alpha!layers!sub_thickness']

        f = pd.read_csv(filtersPath)
        # Get filter properties and respective transmissions from .csv
        filter_property = f[data_string[1]]
        filter_property = filter_property.dropna()
        filter_property = filter_property.values
        transmission = f[data_string[0]]
        transmission = list(transmission.values)

        # N sampling, -1 because of the seperator
        N = int(len(transmission) / len(filter_property)) - 1

        # Define transmission_matrix (#of filters X N+1)
        transmissionmatrix = np.zeros((len(filter_property), N + 1), float)
        filter_metadat = {}

        # Seperator of filter
        zeros = transmission.count(1000)

        # Reshape the transmission data and save filter metadata in a dictionary
        for i in range(len(filter_property)):

            # get transmission data per filter, defined by the seperator
            if i <= zeros - 1:

                # a: min bound, b: max bound
                a = i * (N + 1)
                b = (i + 1) * (N + 1)
                transmissionmatrix[i, :] = (transmission[a:b])

                # Obtain filter metadata, seperator ! and additional seperator is
                # used to distinguish meshnumbers per coordinate i.e. $.
                if '!' in str(filter_property[i]):
                    filter_metadat[r'Filter {}'.format(
                        i + 1)] = filter_property[i].split('!')
                else:
                    pass
            else:
                break

        # Finally remove seperator
        transmissionmatrix = transmissionmatrix[:, 0:N]

        self.filterlibrary = transmissionmatrix

    def loadFilterLibrary(self):
        shapes = ['circle', 'cross', 'cross-hw',
                  'octagon', 'square', 'square-bcc']

        print('Loading filter library...')
        wavelength = np.zeros(1)
        self.structlabels = []
        self.filterlibrary = np.array([])
        for shape in shapes:
            filename = os.path.join(
                module_directory, f'FilterLibrary/FilterLibrary_{shape}_11-08-2022.txt')
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for rowindex, row in enumerate(reader):
                    if rowindex == 0:
                        wavelength = np.float_(row[1:])
                        transmissionprofiles = np.zeros((1, len(wavelength)))
                    elif rowindex == 1:
                        self.structlabels.append(f'{shape}, {row[0]}')
                        transmissionprofiles = np.float_(row[2:])
                    else:
                        self.structlabels.append(f'{shape}, {row[0]}')
                        if np.sum((np.float_(row[2:])) < 1.1) == len(wavelength) and np.sum((np.float_(row[2:])) > -0.0) == len(wavelength):
                            transmissionprofiles = np.vstack(
                                (transmissionprofiles, np.float_(row[2:])))
                        else:
                            transmissionprofiles = np.vstack(
                                (transmissionprofiles, np.zeros((1, len(wavelength)))))

            if self.filterlibrary.size == 0:
                self.filterlibrary = transmissionprofiles
            else:
                self.filterlibrary = np.vstack(
                    (self.filterlibrary, transmissionprofiles))

        # remove wavelengths larger than 1670 nm
        mask = self.wavelength_min <= wavelength * 1e9
        mask = np.logical_and(mask, wavelength * 1e9 < self.wavelength_max)
        self.filterlibrary = np.delete(self.filterlibrary, ~mask, axis=1)
        
        # remove filters with 'ringing' simulation artefact
        lib_fft = np.abs(np.fft.rfft(self.filterlibrary, axis = 1))
        mask =lib_fft[:,21] > 3
        self.filterlibrary = np.delete(self.filterlibrary, mask, axis = 0)
        self.structlabels = np.delete(self.structlabels, mask, axis = 0)
        self.filterlibrarysize = self.filterlibrary.shape[0]
        

    def loadRadianceModel(self):
        print('Loading radiance model...')
        self.radiancemodel = RadianceModel(lambda_min=1625,
                                           lambda_max=1670,
                                           lambda_n=225)

    def filterguess(self):
        smoothness = np.zeros(len(self.structlabels))
        spectral_range = np.linspace(0, 1, self.filterlibrary.shape[1])
        for i in range(len(self.structlabels)):
            p, residuals, rank, singular_values, rcond = np.polyfit(
                spectral_range, self.filterlibrary[i, :], 5, full=True)
            smoothness[i] = residuals

        nfilters = 160
        selection = np.argpartition(smoothness, -nfilters)[-nfilters:]
        selection = np.repeat(selection, 4)
        return selection

    def getTransmissionMatrix(self, selection):
        assert len(selection) == self.instrumentsettings.detector.npxl_alt,\
            f'Selection incorrect size: should be of length {self.instrumentsettings.detector.npxl_alt}'
        transmissionmatrix = self.filterlibrary[selection, :]

        return transmissionmatrix

    def simulateMeasurement(self, selection, nCH4=1500, albedo=0.15, sza = 10, noise=True, n=100, verbose=False):
        self.instrumentsettings.sza = sza
        
        self.transmissionmatrix = self.getTransmissionMatrix(selection)
        radiance, spectral_range = self.radiancemodel.getRadiance(nCH4, albedo, 
                                                                  self.instrumentsettings.sza,
                                                                  self.instrumentsettings.vza)
        self.radiance = radiance

        self.signal = self._getSignal(nCH4, albedo)

        self.noise = np.sqrt(self.signal + self.instrumentsettings.detector.readnoise**2 +
                             self.instrumentsettings.detector.darkcurrent * self.integrationtime)
        self.meanSNR = np.mean(self.signal / self.noise)

        if noise:
            if verbose:
                print(
                    f'Fitting {n} noisy relalizations with nCH4: {nCH4}, albedo: {albedo} and SNR: {self.meanSNR:.1f}')
                starttime = time.perf_counter()

            self.signal_noisy = self.signal[:, None] + \
                self.noise[:, None] * np.random.randn(len(self.signal), n)
            nCH4_fit = np.zeros(n)
            for i in range(n):
                nCH4_fit[i] = self._fitMethane(self.signal_noisy[:, i])
            relative_fitprecision = np.std(nCH4_fit) / nCH4
            relative_fitbias = (nCH4 - np.nanmean(nCH4_fit)) / nCH4

            if verbose:
                endtime = time.perf_counter()
                print(
                    f'Finished methane retrieval for {n} measurements in {(endtime-starttime)*1e3:.1f} ms')
                print(
                    f'Fitted methane: {np.nanmean(nCH4_fit):.1f} ppb +- {np.std(nCH4_fit):.1f} ({relative_fitprecision*1e2:.1f} %), groundtruth: {nCH4:.1f} ppb')

        else:
            nCH4_fit = self._fitMethane(self.signal)
            relative_fitbias = np.abs(nCH4 - nCH4_fit) / nCH4
            relative_fitprecision = 0

        return relative_fitprecision, relative_fitbias

    def _getSignal(self, nCH4, albedo):
        radiance, spectral_range = self.radiancemodel.getRadiance(nCH4, albedo, sza=self.instrumentsettings.sza,
                                                                  vza=self.instrumentsettings.vza)
        signal = np.dot(self.transmissionmatrix, radiance) \
            * self.instrumentsettings.detector.quantumeff \
            * self.integrationtime * self.etendue \
            * (spectral_range[-1] - spectral_range[0]) \
            / len(spectral_range)
        return signal

    def _getFiterror(self, params, tmp, signal_noisy):
        nCH4, albedo = params
        signalfit = self._getSignal(nCH4, albedo)
        return signalfit - signal_noisy

    def _fitMethane(self, signal_noisy):
        p0 = [1000, 0.2]
        fit = optimize.least_squares(
            self._getFiterror, p0, args=([], signal_noisy))

        return fit.x[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.special import gamma
    instrumentsettings = InstrumentSettings()
    instrument = InstrumentSimulator(instrumentsettings)

    print(f'Filter library size: {instrument.filterlibrarysize}')
    #%%
    # Example of how to select filters
    # selection = np.random.randint(0,instrument.filterlibrarysize, size = 640)
    selection = instrument.filterguess()
    # instrument.loadOptimizedFilterset()
    # selection = np.linspace(0,199.99,640, dtype = int)
    
    
    # Methane concentration should be [0,2000], typical: 1500
    nCH4 = 1500

    # low albedo is 0.15, high albedo is 0.7
    albedo = 0.15

    # low sun zenith angle is 70 degrees, high sza = 10
    sza = 10

    # number of noisy realizations
    n = 1000
    #%%
    # retrieve methane
    relative_fitprecision, relative_fitbias,  = instrument.simulateMeasurement(
        selection, nCH4=nCH4, albedo=albedo, sza = sza, n=n, verbose=True)
    print(
        f'bias = {100*relative_fitbias:.1f}%, precision = {100*relative_fitprecision:.2f}%')
    #%%
    selectedfilters = instrument.getTransmissionMatrix(selection)
    fig = plt.figure(dpi=300)
    plt.plot(selectedfilters.T)
