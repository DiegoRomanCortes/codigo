from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
from scipy.linalg import eigh, expm
from scipy.integrate import solve_ivp


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
colors = ['black', 'blue', '#00FFFF', '#00FF00', '#FFFF00', 'orange', 'red', '#FF00FF', 'white']
nodes = [0.0, 1/9 + 0.02, 0.2 + 0.09, 23/63 + 0.06, 12/21, 47/63 - 0.1, 55/63 - 0.15, 0.88, 1.0]
thorlabs = LinearSegmentedColormap.from_list("thorlabs", list(zip(nodes, colors)))

plt.style.use(['science', 'ieee'])

guias = np.zeros((N_GUIAS+1, N_GUIAS, 2))

for i in range(N_GUIAS+1):
    for j in range(N_GUIAS):
        guias[i, j, 0] = 1
        guias[i, j, 1] = 1
        if i == N_GUIAS and j == N_GUIAS - 1:
            guias[i, j, 0] = 0
            guias[i, j, 1] = 0

# setting manually coordinates of impurity at the end of the array
guias[N_GUIAS, 0, 0] = 1
guias[N_GUIAS, 0, 1] = 1

# plt.close('all')

# N_TOTAL = len(guias[:, 0])

# z = np.linspace(0, Z_MAX)
u_0 = np.zeros((N_GUIAS+1, N_GUIAS), dtype=complex)
for i in range(N_GUIAS+1):
    for j in range(N_GUIAS):
        x = -N_GUIAS//4 + i
        y = -N_GUIAS//4 + j
        u_0[i, j] = A * np.exp(-alpha*(x**2 + y**2)) * np.exp(1j*(kx*x + ky*y))

def diff(z, u_vec):
    output = np.zeros((N_GUIAS + 1, N_GUIAS), dtype=complex)
    u_vec = np.reshape(u_vec, output.shape)
    for i in range(N_GUIAS + 1):
        for j in range(N_GUIAS):
            if np.abs(i-N_GUIAS/2) <= N_GUIAS/2 - 2 and np.abs(j-N_GUIAS/2) <= N_GUIAS/2 - 2: 
                output[i, j] = 1j * (u_vec[i+1, j] + u_vec[i-1, j] + u_vec[i, j+1] + u_vec[i, j-1]) * V
            if np.abs(i-N_GUIAS/2) <= 1 and np.abs(j-N_GUIAS/2) <= 1:
                output[i, j] = 1j * ((u_vec[i+1, j] + u_vec[i-1, j] + u_vec[i, j+1] + u_vec[i, j-1]) * V + e * u_vec[N_GUIAS, 0])
                output[N_GUIAS, 0] = 1j*(E*u_vec[N_GUIAS, 0] + e*u_vec[i, j])
    return output.flatten()

z = np.linspace(0, Z_MAX, num=100) 
sol = solve_ivp(diff, (0, Z_MAX), u_0.flatten(), t_eval=z)

frames = 60
for i in range(frames):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.transpose(np.abs(np.reshape(sol.y, (N_GUIAS + 1, N_GUIAS, sol.t.size))[:, :, i*len(z)//frames])**2), cmap=thorlabs, interpolation="kaiser")
    ax.set_aspect('equal')
    ax.set_title(r"$z = {}$".format(np.round(z[i*len(z)//frames]), 3))
    fig.colorbar(im)
    if i < 10:
        fig.savefig("z_0{}.png".format(i))
    else:
        fig.savefig("z_{}.png".format(i))
    plt.close("all")


print()