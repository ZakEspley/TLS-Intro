import numpy as np
import numba
import matplotlib.pyplot as plt

t = np.linspace(0,4,5)
v = np.linspace(0,100,3)
print(t)
print(v)
print()
print(np.outer(t,v))
print()
print(np.outer(v,t))
print()

cos = np.cos(np.outer(v,t))
print(cos)
print()
for v,i in enumerate(["a","b"]):
    print(v,i)

print("Trying")
t = np.array([np.cos,np.sin])
a = np.array([[],[]])
a1 = np.array([[1],[2]])

print(np.concatenate((a,a1),axis=1))


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
# @numba.jit(numba.c16[:,:](numba.float32[:,:],numba.float32,numba.float32,numba.float32,numba.float32),True)
@numba.njit("c16[:,:](f4,f4)")
def H(v,t):
    return -(ω-v)*Sz - Ωx*Sx - 2*Ωz*np.cos(v*t)*Sz - Ωx*(Sp*np.exp(-2j*v*t)+Sm*np.exp(2j*v*t))
@numba.jit(nopython=True)
def EulerTimeEvolve(H, dt, tmax, _v, ω0):
    v = _v / ω0
    t0 = 0
    t = t0
    c10 = 1 +0j
    c20 = 1 - c10**2 + 0j
    cs = np.array([[c10],[c20]])
    # c1 = [c10]
    # c2 = [c20]
    data = np.zeros((int(tmax/dt),2), dtype=np.cdouble)
    print(data)
    # times = []
    # times.append(t0)
    times = np.linspace(0,tmax,int(tmax/dt)+1)
    for i,t in enumerate(times):
        # print(data)
        # print(cs)
        # Euler
        # cs = cs+H/1.0j*t

        cs = cs + (H(v, t)@cs)/ (1.0j)*dt
        data[i]=cs.T
        i+=16
    return data.T

d = EulerTimeEvolve(H,1,5,2,2)
print("Another 1")
print(d)