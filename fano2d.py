from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
from scipy.linalg import eigh, expm


V = 2
E = 0.1
e = 1
Z_MAX = 10
N_GUIAS = 81#11 + center impurity

A = 0.1
alpha = 0.01
kx = 1.55
ky = 1.55
# Escala Thorlabs
colors = ['blue', 'black', 'red']
nodes = [0.0, 0.5, 1.0]
bkr = LinearSegmentedColormap.from_list("thorlabs", list(zip(nodes, colors)))

plt.style.use(['science', 'ieee'])
n = np.arange(-(N_GUIAS-1)//2, (N_GUIAS+1)//2)




guias = np.zeros((N_GUIAS**2+1, 2))
for i in range(N_GUIAS):
    for j in range(N_GUIAS):
        guias[i+N_GUIAS*j, 0] = n[i]
        guias[i+N_GUIAS*j, 1] = n[j]

# setting manually coordinates of impurity at the end of the array
guias[-1, 0] = 0
guias[-1, 1] = 0

# fig0, ax0 = plt.subplots(1, 1)
# ax0.scatter(guias[:-1, 0], guias[:-1, 1], c='red', edgecolors='grey')
# ax0.scatter(guias[-1, 0], guias[-1, 1], c='blue', edgecolors='grey')
# plt.tight_layout()
# fig0.show()
# plt.close('all')

N_TOTAL = len(guias[:, 0])

Cmat = np.zeros((N_TOTAL, N_TOTAL))
for i in range(N_TOTAL):
    Cmat[i, :] = np.sqrt((guias[i, 0] - guias[:, 0])**2 + (guias[i, 1] - guias[:, 1])**2)


Cmat[Cmat > 1] = 0
Cmat[Cmat == 1] = V

Cmat[-1, :] = 0
Cmat[:, -1] = 0
Cmat[(N_GUIAS + 1)*N_GUIAS//2, -1] = e
Cmat[-1, (N_GUIAS + 1)*N_GUIAS//2] = e
Cmat[-1, -1] = E


w, v = eigh(Cmat)



z = np.linspace(0, Z_MAX)
u_0 = np.zeros(N_TOTAL, dtype=complex)
for i in range(N_GUIAS):
    for j in range(N_GUIAS):
        x = -N_GUIAS//2 + i
        y = -N_GUIAS//2 + j
        u_0[i+N_GUIAS*j] = A * np.exp(-alpha*(x**2 + y**2)) * np.exp(1j*(kx*x + ky*y))
u_prop = np.zeros((N_TOTAL, len(z)), dtype=complex)

for idx, i in enumerate(z):
    u_prop[:, idx] = expm(-1j*Cmat*i) @ u_0

fig, ax = plt.subplots(1, 1)

ax.scatter(guias[:, 0], guias[:, 1], c=np.abs(u_prop[:, -1])**2, s=0.1)
ax.set_aspect('equal')
fig.show()


print()