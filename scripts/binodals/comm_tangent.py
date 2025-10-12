import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_v


def get_g0(rho, Dr, eta, w10, nu=0.9, rho0=1, v0=1):
    v = get_v(rho, eta, nu, rho0, v0)
    return np.log(rho * v) + 2 * Dr * w10 / v


if __name__ == "__main__":
    eps = 0.5
    w11 = 0.5 * eps
    w20 = 0.125 * eps ** 2
    w21 = 3 /40 + 1/4 * eps **2
    bar_w2 = 0.5 * (w20 + w21)
    Dr = 1
    eta = 3

    rho_min = 1e-6
    rho_arr = np.linspace(rho_min, 5, 1000)

    g0_arr = np.zeros_like(rho_arr)
    for i, rho in enumerate(rho_arr):
        g0_arr[i] = get_g0(rho, Dr, eta, w11)

    plt.plot(rho_arr, g0_arr)
    plt.show()
    plt.close()