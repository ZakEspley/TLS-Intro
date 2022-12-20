import numpy as np
from scipy.constants import hbar as h
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from matplotlib.colors import LinearSegmentedColormap
from TLS import prefix
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

Ωx = 0.063 # in GHz
Ωz = 0

# Hamiltonian

vmin = 4.8e9
vmax = 5.2e9
n=101
dt = 1e-3
tmax = 200
fig, ax = plt.subplots(1, 1, figsize=(16,9))
colors = plt.cm.gist_rainbow(np.linspace(0,1,n))
# cmap = plt.LinearSegmentedColormap.from_list("jet", colors, N=n)
vs = np.linspace(vmin,vmax,n)
name = "RK4-Excited State2-"
plotC1 = False
plotC2 = True
plotTotal = False
@njit
def main():
    # data = np.empty((2,int(tmax/dt)+1))
    data = []
    for index, _v in enumerate(vs):
        # nd = EulerTimeEvolve(H, dt, tmax, _v, ω0)
        nd = RungeKuttaTimeEvolve(H, dt, tmax, _v, ω0)
        # data = np.concatenate((data,nd), axis=1)
        data.append(nd)
    return data
data = main()
# data = EulerTimeEvolve(H,dt,tmax,5e9,ω0)

times = np.linspace(0,tmax,int(tmax/dt)+1)

for i,d in enumerate(data):
    if plotTotal:
        c1 = np.array(d[0])
        c1 = c1 * c1.conj()
        c2 = np.array(d[1])
        c2 = c2 * c2.conj()
        totalP = c1+c2
        ax.plot(times, totalP, color=colors[i])
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



norm = Normalize(vmin/ω0,vmax/ω0)
sm = plt.cm.ScalarMappable(norm=norm, cmap="gist_rainbow")
fig.colorbar(sm, ax=ax, format="{x:.2f}", label="Driving Frequency (GHz)")
ax.set_title(f"{name} TLS Rotating Frame\n dt={prefix(dt*1e-9,1)}s, v:{prefix(vmin,1)}Hz --> {prefix(vmax,1)}Hz")
# plt.show()
#
fig.savefig(f"{name}-ω={ω}GHz-ϵ={ω}GHz-Ωx={prefix(Ωx*ω0,2)}Hz-ν=({prefix(vmin,1)}-{prefix(vmax,1)}GHz)-dt={prefix(dt*1e-9,1)}s.png")









