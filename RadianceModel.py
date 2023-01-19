# -*- coding: utf-8 -*-
"""
Radiance model of the earth athmosphere.
Based on absorption lines of (traces) gases and the irradiance from the sun a
radiance spectum is computed.


@author: Marijn Siemons
date: 11-11-2022
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from scipy.interpolate import interp1d


module_directory = os.path.dirname(os.path.abspath(__file__))
atmosphere_filename = os.path.join(module_directory, 'Retrieval_Model_data/atmosphere_standard.dat')
irradiance_filename = os.path.join(module_directory, 'Retrieval_Model_data/foresa_solar_irradiance_1600nm_highres.dat')

# %% Functions used in the calculation of the spectrum


def f_young(za):
    """
    Young's function used to approximate optical path in athmosphere'

    Parameters
    ----------
    za : float
        Zenit angle in degrees.

    Returns
    -------
    f : float
        Young correction factor.

    """

    za = math.radians(za)
    f = (1.002432*np.cos(za)**2+0.148386*np.cos(za) + 0.0096467) \
        / (np.cos(za)**3 + 0.149864 * np.cos(za)**2 + 0.0102963 * np.cos(za) + 0.000303978)
    return f


def binning(x, y, xbin):
    '''
    Bins a signal y, with sampling x in binnes defined by xbin.
    xbin defines the center positions of each bin.

    If xbin is outside sampling range x, the value is set to zero.

    Parameters
    ----------
    x : 1D numpy array
        array containing the sampling positions of y. Needs to be linear increasing.
    y : 1D numpy array
        Signal values, should be same size as x.
    xbin : 1D numpy array
        Center position of bins.

    Returns
    -------
    ybin : 1D numpy array
        Integrated values in bins xbin.

    '''

    binsize = xbin[1] - xbin[0]
    ybin = np.zeros(xbin.size)

    for i_bin in range(xbin.size):

        binstart = xbin[i_bin] - binsize/2
        binend = xbin[i_bin] + binsize/2

        # Sum values inside bin
        indices_within_bin = np.nonzero(np.logical_and(x > binstart, x < binend))[0]
        binvalue = np.sum(y[indices_within_bin[0:-1]])

        # determine overlap between other left and right sampling points x
        # and add linear scaled values of these data points.
        if indices_within_bin[0] != 0:
            leftoverlap = (x[indices_within_bin[0]] - binstart) / \
                (x[indices_within_bin[0]] - x[indices_within_bin[0]-1])
            binvalue = binvalue + leftoverlap * y[indices_within_bin[0]-1]
        else:
            leftoverlap = 0

        if indices_within_bin[-1] != x.size-1:
            rightoverlap = (binend - x[indices_within_bin[-1]]) / \
                (x[indices_within_bin[-1]+1] - x[indices_within_bin[-1]])
            binvalue += rightoverlap * y[indices_within_bin[-1]+1]
        else:
            rightoverlap = 0

        ybin[i_bin] = binvalue / (len(indices_within_bin[0:-1]) + leftoverlap + rightoverlap)

    return ybin


class RadianceModel():

    # CONSTANTS
    h_bar = 6.626e-34       # m2 kg / s
    lambda0 = 1.6475e-6     # m
    c = 2.998e8             # m/s
    omega0 = c / lambda0    # 1/s

    M_d = 0.0289652         # kg/mol,
    g = 9.80665             # m/s²
    N_A = 6.02214076e23     # mol⁻¹
    p_0 = 101325            # Pa

    def __init__(self, lambda_min=1625, lambda_max=1670, lambda_n=225):

        # spectral range
        self.spectral_range = np.linspace(lambda_min, lambda_max, lambda_n)

        # Import columns and define number densities
        self.data_abs = np.loadtxt(atmosphere_filename)

        self.nH2O = self.data_abs[:, 3]  # column number density of H2O [cm^-2]
        self.nH2O = np.reshape(self.nH2O, (self.data_abs[:, 3].shape[0], 1))

        self.nCO2 = self.data_abs[:, 4]  # *(8.27e+21/6.023e+23) #column number density of CO2 [cm^-2]
        self.nCO2 = np.reshape(self.nCO2, (self.data_abs[:, 3].shape[0], 1))

        self.nN2O = self.data_abs[:, 5]  # column number density of H2O [cm^-2]
        self.nN2O = np.reshape(self.nN2O, (self.data_abs[:, 3].shape[0], 1))

        self.nCO = self.data_abs[:, 7]  # column number density of CO [cm^-2]
        self.nCO = np.reshape(self.nCO, (self.data_abs[:, 3].shape[0], 1))

        self.nCH4_standard = self.data_abs[:, 8]  # column density of CH4 [cm-2]
        self.nCH4_standard = np.reshape(self.nCH4_standard, (self.data_abs[:, 3].shape[0], 1))

        #                          -- Interpolate irradiance --
        # Import and resample solar radiance

        solar_irradiance_data_object = list(np.transpose(np.loadtxt(irradiance_filename)))
        solar_irradiance_wavevector = np.flip(1.0e7 / solar_irradiance_data_object[0])  # nm^-1 --> nm
        solar_irradiance = np.flip(solar_irradiance_data_object[1])  # sr m2 um
        irradiance = binning(solar_irradiance_wavevector, solar_irradiance, self.spectral_range)

        # Reset units based on photon energy of center wavelength
        # Viable since n ~ 1.0e17 >> 1, allows gaussian distr approx, n>>1 then allows
        # use of center wavelength lambda0.

        # Irradiance_full_range:  W /(sr m2 um) --> ph / (s sr nm m2)
        self.irradiance = irradiance / (1.0e3 * self.h_bar * self.omega0)  # ph / (s sr nm m2)

        #  --- Import and interpolate cross section data ---

        # Import absorption of H20, CO2, N2O, CO, CH4
        sigma_H2O_pl = np.genfromtxt(
            os.path.join(module_directory, 'cross_section_data/cross_section_dataabsorption_cs_H2O_SWIR2.csv'), delimiter=',')
        sigma_CO2_pl = np.genfromtxt(
            os.path.join(module_directory, 'cross_section_data/cross_section_dataabsorption_cs_CO2_SWIR2.csv'), delimiter=',')
        sigma_N2O_pl = np.genfromtxt(
            os.path.join(module_directory, 'cross_section_data/cross_section_dataabsorption_cs_N2O_SWIR2.csv'), delimiter=',')
        sigma_CO_pl = np.genfromtxt(
            os.path.join(module_directory, 'cross_section_data/cross_section_dataabsorption_cs_CO_SWIR2.csv'), delimiter=',')
        sigma_CH4_pl = np.genfromtxt(
            os.path.join(module_directory, 'cross_section_data/cross_section_dataabsorption_cs_CH4_SWIR2.csv'), delimiter=',')

        # Interpolate to spectral range (1625,1670 nm) and convert to dimenstions 300-24

        # Merge all the different absorptions into one array and sample it to be same array as the solar irradiance
        self.sigma = np.zeros((5, lambda_n, sigma_H2O_pl.shape[1]), float)
        for i in range(sigma_H2O_pl.shape[1]):
            self.sigma[0, :, i] = binning(solar_irradiance_wavevector,
                                          np.flip(sigma_H2O_pl[:, i]), self.spectral_range)
            self.sigma[1, :, i] = binning(solar_irradiance_wavevector,
                                          np.flip(sigma_CO2_pl[:, i]), self.spectral_range)
            self.sigma[2, :, i] = binning(solar_irradiance_wavevector,
                                          np.flip(sigma_N2O_pl[:, i]), self.spectral_range)
            self.sigma[3, :, i] = binning(solar_irradiance_wavevector,
                                          np.flip(sigma_CO_pl[:, i]), self.spectral_range)
            self.sigma[4, :, i] = binning(solar_irradiance_wavevector,
                                          np.flip(sigma_CH4_pl[:, i]), self.spectral_range)

    def getRadiance(self, nCH4_ppb, albedo, sza=10, vza=0):
        # Define input concentrtions for methane
        norm_nCH4_standard = np.sum(self.nCH4_standard) * self.M_d * self.g / (self.N_A*self.p_0) * 1e13

        self.nCH4 = nCH4_ppb * self.nCH4_standard / norm_nCH4_standard

        # Calculate vertival absorption
        nLayer = self.nCH4.shape[0]
        tau_vert = self.sigma[0, :, :]@self.nH2O[0:nLayer, 0] + self.sigma[1, :, :]@self.nCO2[0:nLayer, 0]\
            + self.sigma[2, :, :]@self.nN2O[0:nLayer, 0] + self.sigma[3, :, :]@self.nCO[0:nLayer, 0]\
            + (self.sigma[4, :, :]@self.nCH4[0:nLayer, 0])

        # Calculate radiance
        rair = f_young(sza) + f_young(vza)
        consTerm = albedo * np.cos(math.radians(sza)) / np.pi
        tau_lambda = rair * tau_vert
        self.radiance = consTerm * np.exp(-1 * tau_lambda) * self.irradiance  # * 1.0e3 * self.h_bar * self.omega0

        return self.radiance, self.spectral_range


# %% MAIN
if __name__ == "__main__":

    radiancemodel = RadianceModel()
    nCH4 = 2000
    albedo = 0.75
    sza = 10
    spectrum, spectral_range = radiancemodel.getRadiance(nCH4, albedo, sza)
    radiance = spectrum * 1.0e3 * radiancemodel.h_bar * radiancemodel.omega0

    plt.figure(dpi=300)
    plt.plot(spectral_range, spectrum)
    plt.ylabel("Radiance [ph/(s sr nm m2)]")
    plt.xlabel("Wavelength [nm]")
    plt.grid()
    plt.title(f'Methane: {nCH4} ppb ')
