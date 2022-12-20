import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from TLS import *
from scipy.optimize import curve_fit

# Set the default text font size
plt.rc('font', size=16)
# Set the axes title font size
plt.rc('axes', titlesize=16)
# Set the axes labels font size
plt.rc('axes', labelsize=16)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)
# Set the legend font size
plt.rc('legend', fontsize=18)
# Set the font size of the figure title
plt.rc('figure', titlesize=20)
# plt.rc("text", usetex=True)

filename = "TLS-data.h5"
group = "2022-12-20"
runname = "run1"
xunits = "GHz"
title = "Max Excited State Probability"

if __name__ == "__main__":
    file = h5.File(filename, "r")
    data = file[group][runname]
    dt = data.attrs.get("dt")
    tmax = data.attrs.get("tmax")
    vs = data.attrs.get("v")
    n = len(vs)
    ω = data.attrs.get("ω")
    ω0 = data.attrs.get("ω0")
    vmin = np.min(vs) / ω0
    vmax = np.max(vs) / ω0

    fig = plt.figure(figsize=(16, 9))
    ax = fig.subplots()
    ax.set_xlabel(f"Driving Frequency [{xunits}]")
    ax.set_ylabel("Excited State Probability")
    ax.set_title(title)
    maxProbablities = []
    for keys, probabilities in data.items():
        maxP = np.max(np.abs(probabilities[1]) ** 2)
        maxProbablities.append(maxP)
    file.close()

    popt, pcov = curve_fit(Lorentzian, vs / ω0, maxProbablities, p0=[1, 5, 0.5])
    frequencies = np.linspace(vmin - 0.2, vmax + 0.2, 1000)
    Lprobs = Lorentzian(frequencies, popt[0], popt[1], popt[2])
    ax.plot(vs / ω0, maxProbablities, "o")
    ax.plot(frequencies, Lprobs)
    ax.text(4.6, 0.9, "Fitting Function:\n A/pi * 1/2W / ( (f-f0)^2 + (1/2W)^2 )")
    ax.text(4.6, 0.6, f"A = {popt[0]:.2f}\nf0={popt[1]:.2f}GHz\nW={popt[2]:.2f}GHz")
    file.close()
    # plotProbabilities_Sweep(
    #     filename="TLS-data.h5",
    #     group=group,
    #     runname=runname,
    #     title="Frequency Sweep",
    #     xunits="ns",
    #     spacing = 10
    # )
    # plotMaxProbabilities(
    #     filename="TLS-data.h5",
    #     group=group,
    #     runname=runname,
    #     title="Test",
    #     xunits="GHz"
    # )
