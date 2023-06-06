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
group = "2023-03-28"
runname = "run1"
xunits = "GHz"
title = "Rabi Sweep State Probability"

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
    Ωxs = data.attrs.get("Ωx")
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel(f"Driving Frequency [{xunits}]")
    ax.set_ylabel(f"Rabi Frequency [MHz]")
    ax.set_zlabel("Excited State Probability")
    ax.set_title(title)
    v = vs/ω0
    Ωx = Ωxs/ω0
    v, Ωx = np.meshgrid(v,Ωx)
    print(v[:5,:5])
    print(Ωx[:5, :5])
    # for key, rabifrequency in data.items():
    #     dataset = data[key]
    #     rabiF = rabifrequency.attrs.get("Ωx")
    #     Ωxindex = np.where(Ωx == rabiF)[0]
    #     for key2, probabilities in dataset.items():

            
    # maxProbablities = np.empty(v.shape)
    # rabiF = -1
    # rabiCounter = -2
    # tempProbabilities = []
    # for keys, probabilities in data.items():
    #     if data[keys].attrs.get("Ωx") != rabiF:
    #         rabiF = data[keys].attrs.get("Ωx")
    #         rabiCounter += 1
    #         if rabiCounter != -1:
    #             maxProbablities[rabiCounter] = np.array(tempProbabilities)
    #             tempProbabilities = []
    #     maxP = np.max(np.abs(probabilities[1]) ** 2)
    #     tempProbabilities.append(maxP)
    # file.close()
    # ax.plot_surface(v, Ωx, maxProbablities)
    # plt.show()


    plotProbabilities_Sweep(
        filename="TLS-data.h5",
        group=group,
        runname=runname,
        Ωx="Ωx=250.0 M",
        title="Frequency Sweep",
        xunits="ns",
        spacing = 10,
        totalProbability=False,
        excitedstate=True
    )
    # plotMaxProbabilities(
    #     filename="TLS-data.h5",
    #     group=group,
    #     runname=runname,
    #     title="Test",
    #     xunits="GHz"
    # )
    # plotProbabilities_Sweep_Rabi(
    #     filename="TLS-data.h5",
    #     group=group,
    #     runname=runname,
    #     rabiFrequency=6.0e7,
    #     title=f"Frequency Sweep Rabi Frequency=60MHz",
    #     xunits="GHz"
    # )
