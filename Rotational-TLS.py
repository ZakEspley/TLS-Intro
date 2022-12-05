import numpy as np
from scipy.constants import hbar as h
import matplotlib.pyplot as plt
from Useful_Functions import prefix
from numba import njit, vectorize
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

Sp = (Sx + 1j*Sy)/2
Sm = (Sx - 1j*Sy)/2

@njit
def EulerTimeEvolve(H, dt, tmax, _v, ω0):
    v = _v / ω0
    t0 = 0
    t = t0
    c10 = 1+0j
    c20 = (1 - c10**2)+0j
    cs = np.array([[c10],[c20]])
    # c1 = [c10]
    # c2 = [c20]
    data = np.zeros((int(tmax/dt)+1,2),dtype=np.cdouble)
    # times = []
    # times.append(t0)
    times = np.linspace(0,tmax,int(tmax/dt)+1)
    for i,t in enumerate(times):
        # Euler
        cs = cs+(H(v,t)@cs)/(1.0j)*dt
        data[i] = cs.T
    return data.T
# My default frequency
ω0 = 1e9
# Electric Field Frequency
_v = 5e9
v =_v/ω0
# Epsilon
ω = 5
# Angle between the electric field and the moment.
θ = np.pi/2

# Electric Field Amplitude
E0 = 1000

Ωx = 0.063
Ωz = 0

# Hamiltonian
@njit("c16[:,:](f4,f4)")
def H(v,t):
    return -(ω-v)*Sz - Ωx*Sx - 2*Ωz*np.cos(v*t)*Sz - Ωx*(Sp*np.exp(-2j*v*t)+Sm*np.exp(2j*v*t))
print("HERE")
print(H(5.01, 1))
vmin = 4.9e9
vmax = 5.1e9
n=15
dt = 1e-3
tmax = 100
fig, ax = plt.subplots(1, 1)
colors = plt.cm.jet(np.linspace(0,1,n))
vs = np.linspace(vmin,vmax,n)
name = "Rotating-Euler-Method-Test"
@njit
def main():
    # data = np.empty((2,int(tmax/dt)+1))
    data = []
    for index, _v in enumerate(vs):
        nd = EulerTimeEvolve(H, dt, tmax, _v, ω0)
        # data = np.concatenate((data,nd), axis=1)
        data.append(nd)
    return data
data = main()
# data = EulerTimeEvolve(H,dt,tmax,5e9,ω0)
plotC1 = True
plotC2 = True
times = np.linspace(0,tmax,int(tmax/dt)+1)

for i,d in enumerate(data):
    if plotC1:
        c1 = np.array(d[0])
        c1 = c1 * c1.conj()
        c1 = np.real(c1)
        ax.plot(times, c1, color=colors[i])
    if plotC2:
        c2 = np.array(d[1])
        c2 = c2*c2.conj()
        c2 = np.real(c2)
        ax.plot(times, c2, color=colors[i], linestyle="dashed")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Probability Amplitude")
ax.set_title(f"{name} TLS Non-Rotating Frame\n dt={prefix(dt*1e-9,1)}s, v:{prefix(vmin,1)}Hz --> {prefix(vmax,1)}Hz")
# plt.show()
#
fig.savefig(f"{name}-ω={ω}GHz-ϵ={ω}GHz-Ωx={prefix(Ωx*ω0,2)}Hz-ν=({prefix(vmin,1)}-{prefix(vmax,1)}GHz)-dt={prefix(dt*1e-9,1)}s.png")









