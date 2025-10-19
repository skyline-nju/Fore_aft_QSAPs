import numpy as np
import matplotlib.pyplot as plt
from matrix_rho_f1_f2_eq import get_M
from scipy import linalg
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_eta_Pe, get_gamma


def get_rhoB(D, Dc=0.75, A=0.1):
    if D < Dc:
        return A * (Dc - D)
    else:
        return A * Dc


def rho_g_vs_f2():
    eps = 0.5
    w11 = 0.5 * eps
    q_max = 1
    q_arr = np.logspace(-6, np.log10(q_max), 300)

    Dr = 0.1
    eta0 = 3

    f2_arr = np.linspace(0, 0.5, 100)
    rho_arr = np.linspace(0.4, 1.5, 100)
    state = np.zeros((f2_arr.size, rho_arr.size))


    for j, f2 in enumerate(f2_arr):
        for i, rho in enumerate(rho_arr):
            eta, Pe = get_eta_Pe(rho, eta0, Dr)
            gamma = get_gamma(f2, rho, eta0)
            M = get_M(0, q_arr[0], w11, Pe, eta, gamma=gamma)
            sigma_re = np.max(linalg.eigvals(M).real)
            if sigma_re > 0:
                state[j, i] = 1
            else:
                for q in q_arr:
                    M = get_M(q, 0, w11, Pe, eta, gamma=gamma)
                    sigma_re = np.max(linalg.eigvals(M).real)
                    if sigma_re > 0:
                        state[j, i] = 2
                        break
    extent = [rho_arr[0], rho_arr[-1], f2_arr[0], f2_arr[-1]]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.imshow(state, origin="lower", extent=extent, aspect="auto")

    plt.show()
    plt.close()



if __name__ == "__main__":
    eps = 0.5
    w11 = 0.5 * eps
    q_max = 1
    q_arr = np.logspace(-6, np.log10(q_max), 300)

    Dr = 0.1
    eta0 = 3

    f2_arr = np.linspace(0, 0.5, 100)
    rho_arr = np.linspace(0.4, 1.5, 100)
    state = np.zeros((f2_arr.size, rho_arr.size))

    for j, f2 in enumerate(f2_arr):
        for i, rho in enumerate(rho_arr):
            eta, Pe = get_eta_Pe(rho, eta0, Dr)
            gamma = get_gamma(f2, rho, eta0)
            M = get_M(0, q_arr[0], w11, Pe, eta, gamma=gamma)
            sigma_re = np.max(linalg.eigvals(M).real)
            if sigma_re > 0:
                state[j, i] = 1
            else:
                for q in q_arr:
                    M = get_M(q, 0, w11, Pe, eta, gamma=gamma)
                    sigma_re = np.max(linalg.eigvals(M).real)
                    if sigma_re > 0:
                        state[j, i] = 2
                        break
    extent = [rho_arr[0], rho_arr[-1], f2_arr[0], f2_arr[-1]]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.imshow(state, origin="lower", extent=extent, aspect="auto")

    plt.show()
    plt.close()


