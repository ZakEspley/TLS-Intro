import numpy as np
# from scipy.constants import hbar as h
import matplotlib.pyplot as plt
h=1
Sx = 1/2 * np.array([[0,1],
               [1,0]],
              dtype=complex)
Sy = 1/2 * np.array([[0, -1.0j],
               [1.0j, 0]],
              dtype=complex)
Sz = 1/2 * np.array([[1.0, 0],
               [0, -1.0]],
              dtype=complex)
# Electric Field Frequency
v = 1e9
# Potential Energy difference
Δ = 0*h*v
# Tunnelling energy
ΔT = 1*h*v

# Epsilon
ϵ = np.sqrt(Δ**2 + ΔT**2)

# Sine of 2ϕ
s2ϕ = ΔT/ϵ

# Cosine of 2ϕ
c2ϕ = Δ/ϵ

# Tangent of 2ϕ
# t2ϕ = ΔT/Δ

# Angle between the electric field and the moment.
θ = np.pi/4

# Electric Field Amplitude
E0 = 1000



# Moment
p = 1

# Rabi Frequency
Ω0 = p*E0/h

Ωx = Ω0 * np.cos(θ) * s2ϕ
Ωz = Ω0 * np.cos(θ) * c2ϕ
print(Ωx, Ωz)
print(ϵ/h)


# Hamiltonian constant
Hc = -ϵ*Sz
# Hamiltonian Dynamic
Hd = -2*h*Ωz*Sz - 2*h*Ωx*Sx

print(Hd)
print(Hc)


dt = 1e-11
tmax = 100e-9
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
    # k1 = ((Hd @ ( cs*np.cos(v*t) )) + Hc@cs)/(1.0j*h)
    # k2 = ((Hd @ ( (cs+dt*k1/2) * np.cos(v*(t+dt/2)) ) )+ Hc@cs)/(1.0j*h)
    # k3 = ((Hd @ ( (cs+dt*k2/2) * np.cos(v*(t+dt/2)) )) + Hc@cs)/(1.0j*h)
    # k4 = ((Hd @ ( (cs+dt*k3) * np.cos(v*(t+dt)) ) )+ Hc@cs)/(1.0j*h)
    # cs = cs + 1/6 * (k1+2*k2+2*k3+k4)*dt

    # Euler
    m = (( Hd @ (cs *np.cos(v*t)) ) + Hc@cs)/(1.0j*h)
    cs = cs+m*dt
    t = t+dt
    c1.append(cs[0][0])
    c2.append(cs[1][0])
    times.append(t)

c1 = np.array(c1)
c1 = np.sqrt(c1*c1.conj())
c1 = np.real(c1)
# c2 = np.array(c2)
# c2 = np.sqrt(c2*c2.conj())
# c2 = np.real(c2)
fig, ax = plt.subplots(1, 1)
ax.plot(times, c1)
ax.plot(times, c2)
plt.show()







