import numpy as np
from scipy.constants import hbar as h
import matplotlib.pyplot as plt
from TLS import prefix
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

Sx = 1/2 * np.array([[0,1],
               [1,0]],
              dtype=complex)
Sy = 1/2 * np.array([[0, -1.0j],
               [1.0j, 0]],
              dtype=complex)
Sz = 1/2 * np.array([[1.0, 0],
               [0, -1.0]],
              dtype=complex)
# My default frequency
ω0 = 1e9
# Electric Field Frequency
_v = 5e9
# Epsilon
ω = 5
# Angle between the electric field and the moment.
θ = np.pi/2

# Electric Field Amplitude
E0 = 1000

Ωx = 0.063
Ωz = 0

# Hamiltonian constant
Hc = -ω*Sz
# Hamiltonian Dynamic
Hd = -2*Ωz*Sz - 2*Ωx*Sx

def EulerTimeEvolve(dt, tmax, _v, plotC1=True, plotC2=True, color="#000000"):
    v = _v / ω0
    t0 = 0
    t = t0
    c10 = 1
    c20 = 1 - c10**2
    cs = np.array([[c10],[c20]])
    c1 = [c10]
    c2 = [c20]
    times = []
    times.append(t0)

    while t<=tmax:
        # Euler
        m = (( Hd @ (cs *np.cos(v*t)) ) + Hc@cs)/(1.0j)
        cs = cs+m*dt
        t = t+dt
        c1.append(cs[0][0])
        c2.append(cs[1][0])
        times.append(t)
    if plotC1:
        c1 = np.array(c1)
        c1 = np.sqrt(c1 * c1.conj())
        c1 = np.real(c1)
        ax.plot(times, c1, color=color)
    if plotC2:
        c2 = np.array(c2)
        c2 = np.sqrt(c2*c2.conj())
        c2 = np.real(c2)
        ax.plot(times, c2, color=color, linestyle="dashed")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Probability Amplitude")

def RungeKuttaTimeEvolve(dt, tmax, _v, plotC1=True, plotC2=True, color="#000000"):
    v = _v / ω0
    t0 = 0
    t = t0
    c10 = 1
    c20 = 1 - c10**2
    cs = np.array([[c10],[c20]])
    c1 = [c10]
    c2 = [c20]
    times = []
    times.append(t0)

    while t<=tmax:
        # Runge Kutta
        k1 = ((Hd @ ( cs*np.cos(v*t) )) + Hc@cs)/(1.0j)
        k2 = ((Hd @ ( (cs+dt*k1/2) * np.cos(v*(t+dt/2)) ) )+ Hc@cs)/(1.0j)
        k3 = ((Hd @ ( (cs+dt*k2/2) * np.cos(v*(t+dt/2)) )) + Hc@cs)/(1.0j)
        k4 = ((Hd @ ( (cs+dt*k3) * np.cos(v*(t+dt)) ) )+ Hc@cs)/(1.0j)
        cs = cs + 1/6 * (k1+2*k2+2*k3+k4)*dt
        t = t + dt
        c1.append(cs[0][0])
        c2.append(cs[1][0])
        times.append(t)
    if plotC1:
        c1 = np.array(c1)
        c1 = np.sqrt(c1 * c1.conj())
        c1 = np.real(c1)
        ax.plot(times, c1, color=color)
    if plotC2:
        c2 = np.array(c2)
        c2 = np.sqrt(c2*c2.conj())
        c2 = np.real(c2)
        ax.plot(times, c2, color=color, linestyle="dashed")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Probability Amplitude")

def TimeEvolution(dt, tmax, _v, plotC1=True, plotC2=True, color="#000000"):
    v = _v / ω0
    t0 = 0
    t = t0
    c10 = 1
    c20 = 1 - c10 ** 2
    cs = np.array([[c10], [c20]])
    c1 = [c10]
    c2 = [c20]
    times = [t0]
    while t<=tmax:
        # Euler
        cs = ((Hd @ ( cs*np.cos(v*t) )) + Hc@cs)*cs*dt
        t = t+dt
        c1.append(cs[0][0])
        c2.append(cs[1][0])
        times.append(t)

    t = t + dt
    c1.append(cs[0][0])
    c2.append(cs[1][0])
    times.append(t)

    if plotC1:
        c1 = np.array(c1)
        c1 = np.sqrt(c1 * c1.conj())
        c1 = np.real(c1)
        ax.plot(t, c1, color=color)
    if plotC2:
        c2 = np.array(c2)
        c2 = np.sqrt(c2 * c2.conj())
        c2 = np.real(c2)
        ax.plot(t, c2, color=color, linestyle="dashed")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Probability Amplitude")


vmin = 4.0e9
vmax = 6.0e9
n=21
dt = 1e-3
tmax = 100
fig, ax = plt.subplots(1, 1)
colors = plt.cm.jet(np.linspace(0,1,n))
vs = np.linspace(vmin,vmax,n)
name = "TimeEvolution-Method"
for index, _v in enumerate(vs):
    EulerTimeEvolve(dt, tmax, _v, plotC2=False, color=colors[index])
    # RungeKuttaTimeEvolve(dt, tmax, _v, plotC2=False, color=colors[index])
    # TimeEvolution(dt, tmax, _v, plotC2=False, color=colors[index])

ax.set_title(f"{name} TLS Non-Rotating Frame\n dt={prefix(dt*1e9,1)}s, v:{prefix(vmin,1)}Hz --> {prefix(vmax,1)}Hz")
# plt.show()

fig.savefig(f"{name}-ω={ω}GHz-ϵ={ω}GHz-Ωx={prefix(Ωx*ω0,2)}Hz-ν=({prefix(vmin,1)}-{prefix(vmax,1)}GHz)-dt={prefix(dt*1e-9,1)}s.png")








