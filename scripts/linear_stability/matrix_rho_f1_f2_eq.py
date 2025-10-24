import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy import linalg
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_eta_Pe


def get_M2(q, w11, Pe, eta):
    M = np.zeros((2, 2), complex)
    qq = q**2
    M[0, 0] = qq * w11 * eta
    M[0, 1] = -1j * q
    M[1, 0] = -0.5j * q * (1 + eta + 2 * q**2 * eta * w11 * Pe / 16)
    M[1, 1] = - (1/Pe + qq * Pe / 16)
    return M


def get_M3(q, w11, Pe, eta):
    M = np.zeros((3, 3), complex)
    qq = q**2
    D2 = 4 / Pe + qq * Pe / 36
    M[0, 0] = qq * w11 * eta
    M[0, 1] = -1j * q

    M[1, 0] = -0.5j * q * (1+eta)
    M[1, 1] = - 1 / Pe
    M[1, 2] = -0.5j * q

    M[2, 0] = 0.5 * qq * w11 * eta
    M[2, 1] = -0.5j * q
    M[2, 2] = -D2
    return M


def get_M(qx, qy, w11, Pe, eta, gamma=0):
    M = np.zeros((5, 5), complex)
    qq = qx**2 + qy**2
    qq_diff = qx**2 - qy**2
    D2 = 4 / Pe + qq * Pe / 36
    M[0, 0] = qq * w11 * eta + qq_diff * w11 * gamma
    M[0, 1] = -1j * qx
    M[0, 2] = -1j * qy

    M[1, 0] = -0.5j * qx * (1 + eta + gamma)
    M[1, 1] = -1 / Pe
    M[1, 3] = -0.5j * qx
    M[1, 4] = -0.5j * qy

    M[2, 0] = -0.5j * qy * (1 + eta - gamma)
    M[2, 2] = -1 / Pe
    M[2, 3] = 0.5j * qy
    M[2, 4] = -0.5j * qx

    M[3, 0] = -qq * gamma * (Pe / 36 - w11) + 0.5 * qq_diff * w11 * eta
    M[3, 1] = -0.5j * qx
    M[3, 2] = 0.5j * qy
    M[3, 3] = -D2

    M[4, 0] = qx * qy * w11 * eta
    M[4, 1] = -0.5j * qy
    M[4, 2] = -0.5j * qx
    M[4, 4] = -D2
    return M


def PeA_eta_plane():
    w11_0 = 0.25
    PeA_arr = np.linspace(-4, 4, 500)
    PeA_arr = PeA_arr[PeA_arr != 0]
    eta_arr = np.linspace(-5, 5, 500)

    qc = 1
    q_arr = np.logspace(-6, np.log10(qc), 200)

    state = np.zeros((eta_arr.size, PeA_arr.size))
    for j, eta in enumerate(eta_arr):
        for i, PeA in enumerate(PeA_arr):
            qx_min = 1e-5
            qy = 0
            Pe = np.abs((2 * w11_0)/ PeA)
            if PeA < 0:
                w11 = -w11_0
            else:
                w11 = w11_0
            M = get_M(qx_min, qy, w11, Pe, eta)
            sigma_re = np.max(linalg.eigvals(M).real)
            if sigma_re > 0:
                state[j, i] = 1
            else:
                for qx in q_arr:
                    M = get_M(qx, qy, w11, Pe, eta)
                    # M = get_M3(qx, w11, Pe, eta)
                    # M = get_M2(qx, w11, Pe, eta)
                    sigma_re = np.max(linalg.eigvals(M).real)
                    if sigma_re > 0:
                        state[j, i] = 2
                        break

    extent = [PeA_arr[0], PeA_arr[-1], eta_arr[0], eta_arr[-1]]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(state, origin="lower", extent=extent, aspect="auto")

    y1 = eta_arr[eta_arr < 0]
    y2 = eta_arr[eta_arr > 0]
    x1 = (1 + y1) / y1
    x2 = (1 + y2) / y2
    ax.plot(x1, y1, x2, y2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-5, 5)
    plt.show()
    plt.close()



def rho_Dr_plane():
    Dr_arr = np.linspace(0.01, 5, 100)
    rho_arr = np.linspace(0, 1.5, 100)

    eps = 0.5
    eta0 = 3
    q_max = 1
    q_arr = np.logspace(-6, np.log10(q_max), 500)

    w11 = 0.5 * eps
    w22 = w21 = 0

    state = np.zeros((Dr_arr.size, rho_arr.size))
    for j, Dr in enumerate(Dr_arr):
        for i, rho in enumerate(rho_arr):
            eta, Pe = get_eta_Pe(rho, eta0, Dr)
            M = get_M(q_arr[0], 0, w11, Pe, eta)
            sigma_re = np.max(linalg.eigvals(M).real)
            if sigma_re > 0:
                state[j, i] = 1
            else:
                for qx in q_arr:
                    M = get_M(0, qx, w11, Pe, eta)
                    # M = get_M3(qx, w11, Pe, eta)
                    # M = get_M2(qx, w11, Pe, eta)
                    sigma_re = np.max(linalg.eigvals(M).real)
                    if sigma_re > 0:
                        state[j, i] = 2
                        break
    # extent = [rho_arr.min(), rho_arr.max(), Dr_arr.min(), Dr_arr.max()]
    extent = [rho_arr.min(), rho_arr.max(), np.log10(Dr_arr.min()), np.log10(Dr_arr.max())]
    plt.imshow(state, origin="lower", extent=extent, aspect="auto")

    plt.axvline(0.5, linestyle="dashed", color="w")
    plt.axvline(1, linestyle="dashed", color="w")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Dr_arr = np.linspace(0.01, 5, 100)
    # rho_arr = np.linspace(0, 1.5, 100)

    # eps = 0.5
    # eta0 = 3
    # q_max = 1
    # q_arr = np.logspace(-6, np.log10(q_max), 500)

    # w11 = 0.5 * eps
    # w22 = w21 = 0

    # state = np.zeros((Dr_arr.size, rho_arr.size))
    # for j, Dr in enumerate(Dr_arr):
    #     for i, rho in enumerate(rho_arr):
    #         eta, Pe = get_eta_Pe(rho, eta0, Dr)
    #         M = get_M(q_arr[0], 0, w11, Pe, eta)
    #         sigma_re = np.max(linalg.eigvals(M).real)
    #         if sigma_re > 0:
    #             state[j, i] = 1
    #         else:
    #             for qx in q_arr:
    #                 M = get_M(0, qx, w11, Pe, eta)
    #                 # M = get_M3(qx, w11, Pe, eta)
    #                 # M = get_M2(qx, w11, Pe, eta)
    #                 sigma_re = np.max(linalg.eigvals(M).real)
    #                 if sigma_re > 0:
    #                     state[j, i] = 2
    #                     break
    # extent = [rho_arr.min(), rho_arr.max(), Dr_arr.min(), Dr_arr.max()]
    # plt.imshow(state, origin="lower", extent=extent, aspect="auto")

    # plt.axvline(0.5, linestyle="dashed", color="w")
    # plt.axvline(1, linestyle="dashed", color="w")
    # plt.axhline(0.1)
    # plt.show()
    # plt.close()

    q = 1
    w11 = 0.25
    Pe = 1
    eta = 3

    M = get_M3(q, w11, Pe, eta)
    print(M)
