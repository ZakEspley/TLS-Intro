import numpy
import scipy
import numba
import matplotlib.pyplot as plt
import h5py as h5
from TLS import *
import os
import datetime




if __name__ == "__main__":
    # data = h5.File("TLS-data.h5", "a")

    vmin = 4.8e9
    vmax = 5.2e9
    number_of_frequencies = 7
    # frequencies = np.linspace(vmin, vmax, number_of_frequencies)
    #Default Frequency in Hz
    _ω0 = 1e9
    #Resonant Frequency in Hz
    _ω = 5e9
    #Time step in seconds
    _dt = 100e-12
    #Time to run for in seconds
    _tmax = 100e-9
    # Rabi Frequency in Hz
    # _Ωx = np.linspace(10e6, 250e6, 25)
    _Ωx = 63e6
    #AC Stark Shift Frquency in Hz
    _Ωz = 0

    ### Shift times appropriately
    # dt = _ω0*_dt
    # tmax = _ω0*_tmax
    # today = str(datetime.date.today())
    # newgroup = True
    # group = None
    # for groups in data.keys():
    #     if today in groups:
    #         group = data[groups]
    #         lastRun = list(data[groups].keys())[-1]
    #         counter = int(lastRun[3:])+1
    #         newgroup = False
    #         break
    #
    # if newgroup:
    #     group = data.create_group(today, track_order=True)
    #     counter = 1
    # run = group.create_group(f"run{counter}", track_order=True)
    # run.attrs.create('dt', dt)
    # run.attrs.create('tmax', tmax)
    # run.attrs.create('ω0', _ω0)
    # run.attrs.create("ω", _ω)
    # run.attrs.create("Ωx", _Ωx)
    # run.attrs.create("Ωz", _Ωz)
    # run.attrs.create("v", frequencies)
    #
    # # for Ωx in _Ωx:
    # #     for _v in frequencies:
    # #         counter2 = 1
    # #         H = HMaker(_ω0,_v, _ω, Ωx, _Ωz)
    # #         probability_amplitudes = RungeKuttaTimeEvolve(H, dt, tmax)
    # #         ds = run.create_dataset(f"f={prefix(_v,3)}HzΩx={prefix(Ωx,3)}Hz".replace(" ", ""), data=probability_amplitudes, track_order=True, compression="gzip", compression_opts=7)
    # #         ds.attrs.create("Ωx", Ωx)
    # #         ds.attrs.create("v", _v)
    # for _v in frequencies:
    #     counter2 = 1
    #     H = HMaker(_ω0, _v, _ω, _Ωx, _Ωz)
    #     probability_amplitudes = RungeKuttaTimeEvolve(H, dt, tmax)
    #     ds = run.create_dataset(f"f={prefix(_v, 3)}HzΩx={prefix(_Ωx, 3)}Hz".replace(" ", ""),
    #                             data=probability_amplitudes, track_order=True, compression="gzip", compression_opts=7)
    #     ds.attrs.create("Ωx", _Ωx)
    #     ds.attrs.create("v", _v)
    # data.close()
    FrequencySweepSimulation(
        ω0=_ω0,
        ω = _ω,
        dt = _dt,
        tmax=_tmax,
        vmin=vmin,
        vmax=vmax,
        nfrequencies=number_of_frequencies,
        Ωx = _Ωx,
        Ωz = _Ωz,
        compression=7
    )


