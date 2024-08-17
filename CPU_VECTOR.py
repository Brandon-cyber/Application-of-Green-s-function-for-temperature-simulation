import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time


@njit
def fdtr2d_gray(ww, qx, qy, qz, delta, R, vx, tau, CC):
    q = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    vx_tau_q = vx * tau * q
    arctan_term = np.arctan(vx_tau_q / (1.0 + 1.j * ww * tau))
    rr = 1.0 / (vx_tau_q) * arctan_term
    inv_delta = 1.0 / delta
    R2 = R ** 2
    exp_factor = np.exp(-R2 * (qx ** 2 + qy ** 2) / (4 * np.pi))
    ret = R ** 3 * 2.0 * inv_delta / (inv_delta ** 2 + qz ** 2) * exp_factor * rr / (1.0 - rr) / (CC / tau)
    return ret


def compute_deltaT(wlist, qx, qy, qz, delta, R, vx, tau, CC):
    qx_grid, qy_grid, qz_grid = np.meshgrid(qx, qy, qz, indexing='ij')
    deltaT = np.zeros(len(wlist), dtype=np.complex128)

    for i in prange(len(wlist)):
        wi = wlist[i]
        dT = np.sum(fdtr2d_gray(wi, qx_grid, qy_grid, qz_grid, delta, R, vx, tau, CC))
        deltaT[i] = dT

    return deltaT


# Spatial frequency Fourier transform variables
qx = np.linspace(-1e8, 1e8, 1000)
qy = np.linspace(-1e8, 1e8, 1000)
qz = np.linspace(-1e8, 1e8, 1000)

# Gray values from germanium from https://www.science.org/doi/10.1126/sciadv.abg4677
mfp = 4. / 3. * np.sqrt(2) * 4000.0e-9
CC = 1.6e6
vx = 6733. / (2 * np.sqrt(2))
tau = mfp / (2. * vx)
delta = 15.0e-9
R = 3.0e-6

# Temporal frequency Fourier transform variable
wlist = np.logspace(3.0, 15.0, 80)

# Measure the time for compute_deltaT
start_time = time.time()
deltaT = compute_deltaT(wlist, qx, qy, qz, delta, R, vx, tau, CC)
end_time = time.time()

print(f"Time taken for compute_deltaT: {end_time - start_time} seconds")

# Plot phase
plt.semilogx(wlist / (2 * np.pi), np.angle(deltaT) * 180 / np.pi)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase lag (Deg)')
plt.savefig('fdtr2d_gray.pdf')
plt.close()

# Plot amplitude
plt.loglog(wlist / (2 * np.pi), np.abs(deltaT))
plt.savefig('fdtr2d_gray_amp.pdf')
plt.close()
