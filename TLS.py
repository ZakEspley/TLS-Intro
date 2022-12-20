import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import h5py as h5
from scipy.optimize import curve_fit

## Constants and Operators:
Sx = 1 / 2 * np.array([[0, 1],
                       [1, 0]],
                      dtype=complex)
Sy = 1 / 2 * np.array([[0, -1.0j],
                       [1.0j, 0]],
                      dtype=complex)
Sz = 1 / 2 * np.array([[1.0, 0],
                       [0, -1.0]],
                      dtype=complex)

Sp = (Sx + 1j * Sy) / 2
Sm = (Sx - 1j * Sy) / 2


def prefix(x, digits=None):
    prefixes = {
        24: "Y",
        21: "Z",
        18: "E",
        15: "P",
        12: "T",
        9: "G",
        6: "M",
        3: "k",
        0: "",
        -3: "m",
        -6: "µ",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    }
    exponent = np.floor(np.log10(x))
    roundedExp = int(exponent / 3) * 3
    if roundedExp not in prefixes:
        raise ValueError(f"The value: {x} is too large or small for a prefix.")
    prefix = prefixes[roundedExp]
    value = x / (10 ** roundedExp)
    if digits is None:
        return f"{value} {prefix}"
    else:
        return f"{value:0.{digits}f} {prefix}"


@njit
def RungeKuttaTimeEvolve(H: callable, dt: float, tmax: float, groundStateInitial: complex = 1) -> np.ndarray:
    """
    The RungeKuttaTimeEvolve function takes in a Hamiltonian function, a time step, the time you want to run for and the
    initial state of the system. It then uses Runge Kutta to solve for the wavefunction at each time step and returns a
    an array with probabilities amplltudes at each time step.

    :param H:callable: Define the hamiltonian of the system
    :param dt:float: Set the time step
    :param tmax:float: Set the time to which we want to evolve our system
    :param groundStateInitial:complex=1: Set the initial state of the system
    :return: A numpy array of complex numbers, where the first column is ground state probability and the second column
            is the excited state probability amplitude.
    :doc-author: Trelent
    """

    c10 = groundStateInitial + 0j
    c20 = (1 - c10 ** 2) + 0j
    cs = np.array([[c10], [c20]])
    data = np.empty((int(tmax / dt) + 1, 2), dtype=np.cdouble)
    times = np.linspace(0, tmax, int(tmax / dt) + 1)

    for i, t in enumerate(times):
        # Runge Kutta
        k1 = (H(t) @ cs) / (1.0j)
        k2 = (H(t + dt / 2) @ (cs + dt * k1 / 2)) / (1.0j)
        k3 = (H(t + dt / 2) @ (cs + dt * k2 / 2)) / (1.0j)
        k4 = (H(t + dt) @ (cs + dt * k3)) / (1.0j)
        cs = cs + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt
        data[i] = cs.T
    return data.T


@njit
def EulerTimeEvolve(H: callable, dt: float, tmax: float, groundStateInitial: complex = 1) -> np.ndarray:
    """
    The EulerTimeEvolve function takes a Hamiltonian function, a time step, and the total time to evolve
    the system for and the initial condition for the ground state. It will use the simple Euler method to evolve the
    system. It returns an array of the complex amplitudes at each timestep. The first column is ground state probability
    and the second column is the excited state probability amplitude.

    :param H:callable: Define the hamiltonian
    :param dt:float: Set the time step
    :param tmax:float: Set the time to which we want to evolve the system
    :param groundStateInitial:complex=1: Set the initial ground state of the system
    :return: A tuple of two arrays
    :doc-author: Trelent
    """
    c10 = groundStateInitial + 0j
    c20 = (1 - c10 ** 2) + 0j
    cs = np.array([[c10], [c20]])
    data = np.empty((int(tmax / dt) + 1, 2), dtype=np.cdouble)
    times = np.linspace(0, tmax, int(tmax / dt) + 1)
    for i, t in enumerate(times):
        # Euler
        cs = cs + (H(t) @ cs) / (1.0j) * dt
        data[i] = cs.T
    return data.T


def plotProbabilities_Sweep(filename: str, group: str, runname: str, title: str, xunits: str, groundstate=False,
                            excitedstate=True, totalProbability=False, save=False, spacing:int=1):
    """
    The plotProbabilities_Sweep function plots the probabilities of the ground and excited states as a function of time
    for each driving frequency in a sweep. The plot is color-coded according to the driving frequency, following the
    rainbow, lower frequencies are redder, and higher freqeuncies are purple. The dashed lines represent the probability
    of being in the ground state at that time, while the solid line represents being in an excited state.

    :param filename:str: Specify the file name of the hdf5 file
    :param group:str: Select the group in the h5 file
    :param runname:str: Select the data to be plotted
    :param title:str: Set the title of the plot
    :param xunits:str: Change the units of the x-axis
    :param groundstate=False: Show the excited state probability
    :param excitedstate=True: Plot the excited state probability
    :param totalProbability=False: Plot the probability of being in the ground state and excited state separately
    :param save=False: Save the plot as a png file
    :return: None
    :doc-author: Trelent
    """
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

    times = np.linspace(0, tmax, int(tmax / dt) + 1)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, n))

    fig = plt.figure(figsize=(16, 9))
    ax = fig.subplots()
    ax.set_xlabel(f"Time [{xunits}]")
    ax.set_ylabel("Probability State")
    ax.set_title(title)
    norm = Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="gist_rainbow")
    fig.colorbar(sm, ax=ax, format="{x:.2f}", label=f"Driving Frequency [{prefix(ω0,2)[-1:]}Hz]")
    i = 0
    for keys, probabilities in data.items():
        if i%spacing == 0:
            ps = np.abs(probabilities) ** 2
            if groundstate:
                ax.plot(times, ps[0], color=colors[i], linestyle="dashed")
            if excitedstate:
                ax.plot(times, ps[1], color=colors[i])
            if totalProbability:
                ax.plot(times, ps[1]+ps[0], color=colors[i])
        i += 1

    file.close()
    if save:
        fig.savefig(f"{title}.png")
    else:
        plt.show()

def plotMaxProbabilities(filename: str, group: str, runname: str, title: str, xunits: str, groundstate=False,
                            excitedstate=True, totalProbability=False, save=False, spacing:int=1):
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


    popt, pcov = curve_fit(Lorentzian, vs/ω0, maxProbablities, p0=[1, 5, 0.5])
    frequencies = np.linspace(vmin - 0.2, vmax + 0.2, 1000)
    Lprobs = Lorentzian(frequencies, popt[0], popt[1], popt[2])
    ax.plot(vs / ω0, maxProbablities, "o")
    ax.plot(frequencies, Lprobs)
    ax.text(4.6,0.9,"Fitting Function:\n A/pi * 1/2W / ( (f-f0)^2 + (1/2W)^2 )")
    ax.text(4.6,0.6, f"A = {popt[0]:.2f}\nf0={popt[1]:.2f}GHz\nW={popt[2]:.2f}GHz")
    file.close()
    if save:
        fig.savefig(f"{title}.png")
    else:
        plt.show()

def Lorentzian(x,A,x0,g):
    hg = g/2
    return A/np.pi * hg/ ( (x-x0)**2 + hg**2 )
