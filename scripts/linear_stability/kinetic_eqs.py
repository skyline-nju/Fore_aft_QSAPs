import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy import linalg
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_eta_Pe


def get_M4(q, w11, Pe, eta):
    qq = q**2
    M = np.zeros((4, 4), complex)
    M[0, 0] = qq * w11 * eta
    M[0, 1] = -1j * q
    M[1, 0] = -0.5j * q * (1+eta)
    M[1, 1] = -1 / Pe
    M[1, 2] = -0.5j * q
    M[2, 0] = 0.5 * qq * w11 * eta
    M[2, 1] = -0.5j * q
    M[2, 2] = -4 / Pe
    M[2, 3] = -0.5j * q
    M[3, 2] = -0.5j * q
    M[3, 3] = -9/Pe - qq * Pe / 4 / 4**2
    # M[3, 3] = -9./Pe
    return M


def get_M(q, K, w11, Pe, eta, D_K_on=True, Dt=0):
    qq = q**2
    M0 = np.zeros((K+1, K+1), complex)
    M0[0, 0] = qq * w11 * eta
    M0[0, 1] = -0.5j * q
    M0[1, 0] = -0.5j * q * eta
    M0[2, 0] = 0.5 * qq * w11 * eta

    diag_arr = -1 / Pe * np.array([i**2 for i in range(K+1)]) - qq * Dt * np.ones(K+1)
    if D_K_on:
        diag_arr[-1] -= 0.25 * qq * Pe / (K+1)**2 
    M = M0 - 0.5j * q * ( np.eye(K+1, k=1) + np.eye(K+1, k=-1)) + np.diag(diag_arr)
    return M


def get_M0(q, K, w11, eta):
    qq = q**2
    M0 = np.zeros((K+1, K+1), complex)
    M0[0, 0] = qq * w11 * eta
    M0[0, 1] = -0.5j * q
    M0[1, 0] = -0.5j * q * eta
    M0[2, 0] = 0.5 * qq * w11 * eta

    M = M0 - 0.5j * q * ( np.eye(K+1, k=1) + np.eye(K+1, k=-1))
    return M


def get_SOI_line_eps_vs_Dr(eta=3, K=3, qc=1, D_K_on=False):
    eps_arr = np.logspace(-3, 0, 2000)
    Dr_arr = np.logspace(-3, 0, 200)

    eps_c = np.zeros_like(Dr_arr)
    last_idx = 0
    for j, Dr in enumerate(Dr_arr):
        Pe = 1 / Dr
        for i, eps in enumerate(eps_arr[last_idx:]):
            w11 = eps * 0.5
            M = get_M(qc, K, w11, Pe, eta, D_K_on=D_K_on)
            sigma_re = np.max(linalg.eigvals(M).real)
            if sigma_re > 0:
                last_idx = i
                eps_c[j] = eps
                break
    return eps_c, Dr_arr


def eps_Dr_plane(eta = 3, K=2, D_K_on=True, Dt=0):
    eps_arr = np.logspace(-3, 0, 100)
    Dr_arr = np.logspace(-3, 0, 100)

    state = np.zeros((Dr_arr.size, eps_arr.size))
    q_thresh = np.zeros_like(state)
    qc = 1
    q_arr = np.logspace(-6, np.log10(qc), 100)

    for j, Dr in enumerate(Dr_arr):
        Pe = 1/Dr
        for i, eps in enumerate(eps_arr):
            w11 = eps * 0.5
            qx_min = q_arr[0]
            M = get_M(qx_min, K, w11, Pe, eta, D_K_on, Dt)
            sigma_re = np.max(linalg.eigvals(M).real)
            if sigma_re > 0:
                state[j, i] = 1
            else:
                for q in q_arr[1:]:
                    # M = get_M4(q, w11, Pe, eta)
                    M = get_M(q, K, w11, Pe, eta, D_K_on, Dt)
                    sigma_re = np.max(linalg.eigvals(M).real)
                    if sigma_re > 0:
                        state[j, i] = 2
                        q_thresh[j, i] = q
                        break

    extent = [np.log10(Dr_arr[0]), np.log10(Dr_arr[-1]), np.log10(eps_arr[0]), np.log10(eps_arr[-1])]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(state, origin="lower", extent=extent, aspect="auto")
    ax2.imshow(q_thresh, origin="lower", extent=extent, aspect="auto")

    plt.show()
    plt.close()

if __name__ == "__main__":
    # eps_Dr_plane(K=3, D_K_on=False, Dt=0)
    eps, Dr = get_SOI_line_eps_vs_Dr(K=3)

    plt.plot(eps, Dr)
    plt.show()
    plt.close()


