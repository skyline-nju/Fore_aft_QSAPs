import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_v, get_rho


def get_g0(rho, eta, Dr, w10, nu=0.9, rho0=1, v0=1):
    v = get_v(rho, eta, nu, rho0, v0)
    return np.log(rho * v) + 2 * Dr * w10 / v


def get_rho_thresh(eta, Dr, w10, nu=0.9, rho0=1, v0=1):
    vc = 2 * Dr * w10
    if v0-nu <= vc <= v0+nu:
        rho_c = get_rho(vc, eta, nu, rho0, v0)
        return rho_c
    else:
        return None


if __name__ == "__main__":
    rho_min = 1e-9
    eps = 0.3
    w10 = 0.5 * eps

    eta = 3
    Dr = 3

    rho_arr = np.linspace(rho_min, 5, 10000)
    g0_arr = get_g0(rho_arr, eta, Dr, w10)

    plt.plot(rho_arr, g0_arr)

    rho_c = get_rho_thresh(eta, Dr, w10)
    if rho_c is not None:
        print(rho_c)
        plt.axvline(rho_c, linestyle="--")

    plt.show()
    plt.close()