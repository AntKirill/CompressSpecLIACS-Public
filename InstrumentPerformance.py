# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:19:28 2022

@author: marijns
"""
from instrumentsimulator import InstrumentSettings, InstrumentSimulator
import matplotlib.pyplot as plt
from scipy.special import gamma
import numpy as np

instrumentsettings = InstrumentSettings()
instrumentsettings.aperture_act = 60e-3
instrumentsettings.aperture_alt = 60e-3
instrument = InstrumentSimulator(instrumentsettings)
selection = instrument.filterguess()
# instrument.loadOptimizedFilterset()
# selection = np.linspace(0,199.99,640, dtype = int)

# number of noisy realizations
n = 500

# %%
nCH4 = 1500
albedo_range = np.linspace(0.15, 0.75, 15)
sza_range = np.array([10, 30, 50, 70])
fitprecision = np.zeros((len(sza_range), len(albedo_range)))
for j, sza in enumerate(sza_range):
    for i, albedo in enumerate(albedo_range):
        # retrieve methane
        relative_fitprecision, relative_fitbias,  = instrument.simulateMeasurement(
            selection, nCH4=nCH4, albedo=albedo, sza=sza, n=n, verbose=False)

        fitprecision[j, i] = relative_fitprecision

# %%

fig = plt.figure(dpi=300)
for j, sza in enumerate(sza_range):
    plt.plot(albedo_range, 100*fitprecision[j, :], label=f'sza: {sza}')
plt.legend()
plt.xlabel('albedo')
plt.ylabel('relative precision (%)')
plt.title(f'Methane retrieval, {nCH4} ppb')
plt.grid('major')
plt.ylim([0, 2])

# %%
n = 500
nCH4_range = np.linspace(1000, 3000, 20)
snrconfig = [['high', 10, 0.75], ['low', 70, 0.15]]
nCH4_std = np.zeros((len(snrconfig), len(nCH4_range)))
nCH4_fit = np.zeros((len(snrconfig), len(nCH4_range)))
for j, config in enumerate(snrconfig):
    _, sza, albedo = config
    for i, nCH4_i in enumerate(nCH4_range):
        # retrieve methane
        relative_fitprecision, relative_fitbias,  = instrument.simulateMeasurement(
            selection, nCH4=nCH4_i, albedo=albedo, sza=sza, n=n, verbose=False)

        nCH4_fit[j, i] = relative_fitbias * nCH4_i + nCH4_i
        nCH4_std[j, i] = relative_fitprecision * nCH4_i

# %%

fig = plt.figure(dpi=300, figsize=(4.5, 3.7))
j = 1
config = snrconfig[j]
plt.errorbar(nCH4_range, nCH4_fit[j, :], nCH4_std[j, :],
             label=f'{config[0]} SNR, sza: {config[1]}, albedo: {config[2]}')
j = 0
config = snrconfig[j]
plt.errorbar(nCH4_range, nCH4_fit[j, :], nCH4_std[j, :],
             label=f'{config[0]} SNR, sza: {config[1]}, albedo: {config[2]}')
plt.plot([1000, 2000], [1000, 2000], 'k--')
plt.legend()
plt.xlabel('methane concentration (ppb)')
plt.ylabel('retrieved concentration')
plt.title(f'Methane retrieval')
plt.grid('major')
ax = plt.gca()
# ax.set_aspect('equal')
