# -*- coding: utf-8 -*-
"""
Instrument Simulator for spectrometer instrument based on photonic crystals

@author: Marijn Siemons
date: 11-11-2022

"""

import numpy as np
from dataclasses import dataclass
from RadianceModel import RadianceModel
import os
import csv
from scipy import optimize
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
    spatialbinning: int = 5


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

    def loadOldOptimizedFilterset(self):
        """
        Loads the old optimized filter set, based on the code of Menno Hagenaar
        DO NOT USE, unless for comparison.

        Returns
        -------
        None.

        """
        import pandas as pd

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
        """
        Loads the filter library.

        Returns
        -------
        None.

        """

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
                        if np.sum((np.float_(row[2:])) < 1.1) == len(wavelength) and np.sum((np.float_(row[2:])) > -0.0) == len(wavelength):
                            self.structlabels.append(f'{shape}, {row[0]}')
                            transmissionprofiles = np.vstack(
                                (transmissionprofiles, np.float_(row[2:])))

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
        lib_fft = np.abs(np.fft.rfft(self.filterlibrary, axis=1))
        mask = lib_fft[:, 21] > 3
        self.filterlibrary = np.delete(self.filterlibrary, mask, axis=0)
        self.structlabels = np.delete(self.structlabels, mask, axis=0)

        self.filterlibrarysize = self.filterlibrary.shape[0]

    def loadRadianceModel(self):
        """
        Initializes the Radiance model and athmosphere data.

        Returns
        -------
        None.

        """
        print('Loading radiance model...')
        self.radiancemodel = RadianceModel(lambda_min=1625,
                                           lambda_max=1670,
                                           lambda_n=225)

    def filterguess(self):
        """
        Initial guess of a filter set, based on the second moment of the FT of
        the transmission profiles. 160 filters are selected and repeated 4 times.

        Returns
        -------
        selection : array
            list of the indexes of the selected filters

        """
        nfilters = 160
        Tfft = np.fft.rfft(self.filterlibrary, axis=1)
        wavelength_fft = np.fft.rfftfreq(self.wavelength_n, 1)
        Tfft_width = np.sum(np.abs(Tfft)**2 * (wavelength_fft[None, :]/1e9)**2, axis=1)
        selection = np.argpartition(Tfft_width, -nfilters)[-nfilters:]
        selection = np.repeat(selection, 4)

        return selection

    def getDistanceFilterSet(self, filterset1_idx, filterset2_idx, method):
        """
        Computes a distance/simularity of two filter sets using 2 methods:
            2. The variance in the ratio of transmission of the methane absorption lines of the filter set
            3. The sum of the 2nd moment of fourier transforms of the filter set

        Parameters
        ----------
        filterset1_idx : list(int)
            List of indexes for filterset 1.
        filterset1_idx : list(int)
            List of indexes for filterset 2.
        method : int
            Method index. Should be 1 or 2.

        Returns
        -------
        distance : float
            distance metric between the two filter sets. Always positive.

        """
        assert method in [2, 3], 'Method should be in [2, 3]'
        assert len(filterset1_idx) == self.instrumentsettings.detector.npxl_alt,\
            f'Selection filterset 1 incorrect size: should be of length {self.instrumentsettings.detector.npxl_alt}'
        assert len(filterset2_idx) == self.instrumentsettings.detector.npxl_alt,\
            f'Selection filterset 2 incorrect size: should be of length {self.instrumentsettings.detector.npxl_alt}'

        if method == 1:
            transmissionmatrix_1 = self.getTransmissionMatrix(filterset1_idx)
            transmissionmatrix_2 = self.getTransmissionMatrix(filterset2_idx)

            # Get mean absorption of methane over the athmosphere
            absorption = np.mean(self.radiancemodel.sigma[4, :, :], axis=1)
            absorption = absorption / np.max(absorption)

            signal_ratio_1 = transmissionmatrix_1 @ absorption / np.sum(transmissionmatrix_1, axis=1)
            signal_ratio_2 = transmissionmatrix_2 @ absorption / np.sum(transmissionmatrix_2, axis=1)

            signal_var_1 = np.var(signal_ratio_1)
            signal_var_2 = np.var(signal_ratio_2)

            distance = np.abs(signal_var_1 - signal_var_2)

        elif method == 2:
            transmissionmatrix_1 = self.getTransmissionMatrix(filterset1_idx)
            transmissionmatrix_2 = self.getTransmissionMatrix(filterset2_idx)
            wavelength_fft = np.fft.rfftfreq(self.wavelength_n, 1)

            transmissionmatrix_fft_1 = np.fft.rfft(transmissionmatrix_1, axis=1)
            transmissionmatrix_fft_2 = np.fft.rfft(transmissionmatrix_2, axis=1)

            fft_width_1 = np.sum(np.abs(transmissionmatrix_fft_1)**2 * (wavelength_fft[None, :])**2)
            fft_width_2 = np.sum(np.abs(transmissionmatrix_fft_2)**2 * (wavelength_fft[None, :])**2)

            distance = np.abs(fft_width_1 - fft_width_2)
        else:
            distance = None

        return distance

    def getDistanceFilter(self, filter_index1, filter_index2, method):
        """
        Computes a distance/simularity of two filters using 3 methods:
            1. maximum normalized cross correlation
            2. ratio of transmission of the methane absorption lines
            3. 2nd moment of fourier transforms

        Parameters
        ----------
        filter_index1 : int
            Index of filter 1.
        filter_index2 : int
            Index of filter 2.
        method : int
            Method index. Should be 1, 2, or 3.

        Returns
        -------
        distance : float
            distance metric between the two filters. Always positive.

        """
        assert -1 < filter_index1 < self.filterlibrarysize, 'Filter index 1 should be positive and not bigger than the filter library size'
        assert -1 < filter_index2 < self.filterlibrarysize, 'Filter index 2 should be positive and not bigger than the filter library size'
        assert method in [1, 2, 3], 'Method should be in [1, 2, 3]'

        if method == 1:
            # Get filters
            filter1 = instrument.filterlibrary[filter_index1, :]
            filter1 = filter1 / np.sum(filter1)
            filter2 = instrument.filterlibrary[filter_index2, :]
            filter2 = filter2 / np.sum(filter2)

            # Compute cross correlation
            crosscor = np.convolve(filter1, filter2, mode='same')

            # normalize
            # norm = np.convolve(np.ones(len(filter1)), np.ones(len(filter2)), mode='same') / len(filter1)
            # crosscor = crosscor / norm

            # Compute distance
            distance = 1 - 1 / np.max(crosscor)

        elif method == 2:
            # Get mean absorption of methane over the athmosphere
            absorption = np.mean(self.radiancemodel.sigma[4, :, :], axis=1)
            absorption = absorption / np.max(absorption)

            # get filters
            filter1 = self.filterlibrary[filter_index1, :]
            filter2 = self.filterlibrary[filter_index2, :]

            # transmission of methane absorption lines
            filter1_absoption_ratio = np.sum(filter1 * absorption) / np.sum(filter1)
            filter2_absoption_ratio = np.sum(filter2 * absorption) / np.sum(filter2)

            # Compute distance
            distance = np.abs(filter1_absoption_ratio - filter2_absoption_ratio)

        elif method == 3:
            # get fft of filters
            filter1_fft = np.abs(np.fft.rfft(self.filterlibrary[filter_index1, :]))**2
            filter2_fft = np.abs(np.fft.rfft(self.filterlibrary[filter_index2, :]))**2

            # 2nd Moment
            wavelength_fft = np.fft.rfftfreq(self.wavelength_n, 1)
            filter1_fft_2ndmom = np.sum(filter1_fft * wavelength_fft[None, :]**2)
            filter2_fft_2ndmom = np.sum(filter2_fft * wavelength_fft[None, :]**2)

            # Compute distance
            distance = np.abs(filter1_fft_2ndmom - filter2_fft_2ndmom)

        else:
            distance = None

        return distance

    def getTransmissionMatrix(self, selection):
        """
        Creates the transmission based on the list of indexes (selection)

        Parameters
        ----------
        selection : array
            list of indexes of the selected filters.

        Returns
        -------
        transmissionmatrix : 2D array
            The transmission matrix.

        """
        assert len(selection) == self.instrumentsettings.detector.npxl_alt,\
            f'Selection incorrect size: should be of length {self.instrumentsettings.detector.npxl_alt}'
        transmissionmatrix = self.filterlibrary[selection, :]

        return transmissionmatrix

    def simulateMeasurement(self, selection, nCH4=1500, albedo=0.15, sza=10, noise=True, n=100, verbose=False):
        """
        Simulates a series of measurements for the given condition.

        Parameters
        ----------
        selection : array
            list of choosen filters.
        nCH4 : float, optional
            Methane concentration in ppb. The default is 1500.
        albedo : float, optional
            The albedo, should be between 0.15 and 0.75. The default is 0.15.
        sza : float, optional
            The sun zenith angle in degrees, should be between 10 and 70.
            The default is 10.
        noise : boolean, optional
            Flag to include in noise in measurements. The default is True.
        n : int, optional
            Number of noisy measurements. The default is 100.
        verbose : bool, optional
            Flag to plot information. The default is False.

        Returns
        -------
        relative_fitprecision : float
            The relative methane fit precision .
        relative_fitbias : float
            The relative bias in the estimated methane concentration.

        """

        self.instrumentsettings.sza = sza

        self.transmissionmatrix = self.getTransmissionMatrix(selection)
        radiance, spectral_range = self.radiancemodel.getRadiance(nCH4, albedo,
                                                                  self.instrumentsettings.sza,
                                                                  self.instrumentsettings.vza)
        self.radiance = radiance
        self.spectral_range = spectral_range

        self.signal = self._getSignal(nCH4, albedo) * self.instrumentsettings.spatialbinning

        self.noise = np.sqrt(self.instrumentsettings.spatialbinning*(self.signal + self.instrumentsettings.detector.readnoise**2 +
                             self.instrumentsettings.detector.darkcurrent * self.integrationtime))
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
        """
        Generates a camera signal

        Parameters
        ----------
        nCH4 : float
            methane concentation in ppb.
        albedo : float
            Albedo of the scene.

        Returns
        -------
        signal : array
            Signal intensity as measured by the camera.

        """
        radiance, spectral_range = self.radiancemodel.getRadiance(nCH4, albedo, sza=self.instrumentsettings.sza,
                                                                  vza=self.instrumentsettings.vza)
        signal = np.dot(self.transmissionmatrix, radiance) \
            * self.instrumentsettings.detector.quantumeff \
            * self.integrationtime * self.etendue \
            * (spectral_range[-1] - spectral_range[0]) \
            / len(spectral_range)
        return signal

    def _getFiterror(self, params, tmp, signal_noisy):
        """
        Returns the fit error.

        Parameters
        ----------
        params : (nCH4, albedo)
            Tuple containing the (estimated) methane concentration and albedo.
        tmp : empty
            Not used.
        signal_noisy : array
            Noise corrupted signal measured by the camera.

        Returns
        -------
        fit error : float
            fit error between the estimated signal and the noise corrupted signal.

        """
        nCH4, albedo = params
        signalfit = self._getSignal(nCH4, albedo)
        return signalfit - signal_noisy

    def _fitMethane(self, signal_noisy):
        """
        Performs the methane fit.

        Parameters
        ----------
        signal_noisy : array
            Noise corrupted signal measured by the camera.

        Returns
        -------
        nCH4 : float
            fitted methane concentration in ppb.

        """
        p0 = [2000, 0.2]
        fit = optimize.least_squares(
            self._getFiterror, p0, args=([], signal_noisy))

        return fit.x[0]


# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    instrumentsettings = InstrumentSettings()
    instrument = InstrumentSimulator(instrumentsettings)

    print(f'Filter library size: {instrument.filterlibrarysize}')

    # %%
    # Example of how to select filters
    # selection = np.random.randint(0,instrument.filterlibrarysize, size = 640)
    selection = instrument.filterguess()

    # Methane concentration should be [1500,3000], typical: 1800
    nCH4 = 1800

    # low albedo is 0.15, high albedo is 0.7
    albedo = 0.15

    # low sun zenith angle is 70 degrees, high sza = 10
    sza = 70

    # number of noisy realizations
    n = 1000

    # retrieve methane
    relative_fitprecision, relative_fitbias,  = instrument.simulateMeasurement(
        selection, nCH4=nCH4, albedo=albedo, sza=sza, n=n, verbose=True)
    print(
        f'bias = {100*relative_fitbias:.1f}%, precision = {100*relative_fitprecision:.2f}%')

    selectedfilters = instrument.getTransmissionMatrix(selection)
    fig = plt.figure(dpi=300)
    plt.plot(instrument.spectral_range, selectedfilters.T[:, ::4])
    plt.xlabel('wavelength (nm)')
    plt.ylabel('transmission')
    plt.title('Transmission profiles')

    # %% creates distance matrix
    N = 1000
    distance = np.zeros((3, N, N))

    for i in range(N):
        ind1 = i
        for j in np.arange(i+1, N):
            ind2 = j
            for method in [1, 2, 3]:
                distance[method-1, i, j] = instrument.getDistance(ind1, ind2, method)

    # %%
    filter_index = 0
    title = 'Inv peak cross correlation'
    plt.figure(dpi=300)
    plt.hist(distance[filter_index].flatten(), bins=100, range=[0, 3000])
    plt.title(title)
    plt.xlabel('metric')
    plt.ylabel('number of relations between filters')

    plt.figure(dpi=300)
    plt.imshow(np.log(distance[filter_index]))
    plt.colorbar()
    plt.title(title + '(log)')

    # %%
    plt.plot(instrument.spectral_range, instrument.filterlibrary[163, :])
