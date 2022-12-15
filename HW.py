import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1,50,11)
y = np.linspace(-50,50,21)

negx,negy = np.meshgrid(-x,y)
posx,posy = np.meshgrid(x,y)
# v = 10+np.zeros((1,11))
# u = y/2
pdu = posy/2
pdv = (posx-posx)-10
ndu = negy/2
ndv = (negx-negx)+10
# posdir = np.meshgrid(u,v)
# negdir = np.meshgrid(u,-v)
# print(posdir)
# print(neg)
fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.quiver(negx,negy,ndu,ndv)
ax.quiver(posx,posy,pdu,pdv)
ax.plot([0,0],[-50,50], "b-")
ax.plot([-50,50],[0,0], "b-")
plt.show()

