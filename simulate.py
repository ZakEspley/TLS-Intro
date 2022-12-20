import numpy
import scipy
import numba
import matplotlib.pyplot as plt
import h5py as h5
from TLS import *
import os
import datetime






def HMaker(_ω0:float, _v:float, _ω:float, _Ωx:float, _Ωz:float) -> callable:
    """
    The HMaker function returns a Hamiltonian that takes one argument, t. This can then be fed into the EulerTimeEvolve
    or the RungeKuttaTimeEvolve functions from TLS.

    :param _ω0:float: Scale the frequency to the correct units
    :param _v:float: Define the frequency of teh external field in Hz
    :param _ω:float_Ωx:float: Define the rabi frequency in Hz
    :param _Ωz:float)-&gt;: Define the AC Stark Shift Frequency
    :return: A function the hamiltonian, t
    :doc-author: Trelent
    """
    v = _v/_ω0
    ω = _ω/_ω0
    Ωx = _Ωx/_ω0
    Ωz = _Ωz/_ω0

    @njit("c16[:,:](f4)")
    def H(t):
        return -(ω - v) * Sz - Ωx * Sx - 2 * Ωz * np.cos(v * t) * Sz - Ωx * (
                Sp * np.exp(-2j * v * t) + Sm * np.exp(2j * v * t))
    return H


if __name__ == "__main__":
    data = h5.File("TLS-data.h5", "a")

    vmin = 4.8e9
    vmax = 5.2e9
    number_of_frequencies = 101
    frequencies = np.linspace(vmin, vmax, number_of_frequencies)
    #Default Frequency in Hz
    _ω0 = 1e9
    #Resonant Frequency in Hz
    _ω = 5e9
    #Time step in seconds
    _dt = 1e-12
    #Time to run for in seconds
    _tmax = 200e-9
    # Rabi Frequency in Hz
    _Ωx = 63e6
    #AC Stark Shift Frquency in Hz
    _Ωz = 0

    ### Shift times appropriately
    dt = _ω0*_dt
    tmax = _ω0*_tmax
    today = str(datetime.date.today())
    newgroup = True
    group = None
    for groups in data.keys():
        if today in groups:
            group = data[groups]
            lastRun = list(data[groups].keys())[-1]
            counter = int(lastRun[3:])+1
            newgroup = False
            break

    if newgroup:
        group = data.create_group(today, track_order=True)
        counter = 1
    run = group.create_group(f"run{counter}", track_order=True)
    run.attrs.create('dt', dt)
    run.attrs.create('tmax', tmax)
    run.attrs.create('ω0', _ω0)
    run.attrs.create("ω", _ω)
    run.attrs.create("Ωx", _Ωx)
    run.attrs.create("Ωz", _Ωz)
    run.attrs.create("v", frequencies)

    for _v in frequencies:
        counter2 = 1
        H = HMaker(_ω0,_v, _ω, _Ωx, _Ωz)
        probability_amplitudes = RungeKuttaTimeEvolve(H, dt, tmax)
        run.create_dataset(f"f={prefix(_v,3)}Hz".replace(" ", ""), data=probability_amplitudes, track_order=True)

    data.close()


