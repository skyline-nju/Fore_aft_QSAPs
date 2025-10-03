import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_v_dv
from chemical_potential import get_g0


def get_varphi_prime(rho, Dr, eta, w10, w2_bar, nu=0.9, rho0=1, v0=1):
    def func_int(rho_hat, Dr, eta, w10, nu, rho0, v0):
        v, dv = get_v_dv(rho_hat, eta, nu, rho0, v0)
        return dv / (v - 2 * Dr * w10)

    v, v_prime = get_v_dv(rho, eta, nu, rho0, v0)
    int1, err = integrate.quad(func_int, 0, rho, args=(Dr, eta, w10, nu, rho0, v0))
    I_rho = (1 - 3 * w10**2/(2*w2_bar)) * np.log(v_prime) - 2 * np.log(v) + int1
    # I_rho = (1 - 3 * w10**2/(2*w2_bar)) * np.log(v_prime) + 1 * np.log(v) 
    return np.exp(-I_rho)

def get_h0(rho, Dr, eta, w10, w2_bar, nu=0.9, rho0=1, v0=1, rho_min=1e-6):
    def func_int(y, g0_rho, Dr, eta, w10, w2_bar, nu, rho0, v0):
        g0_y = get_g0(y, eta, Dr, w10, nu, rho0, v0)
        varphi_prime = get_varphi_prime(y, Dr, eta, w10, w2_bar, nu, rho0, v0)
        return (g0_rho - g0_y) * varphi_prime
    
    g0_rho = get_g0(rho, eta, Dr, w10, nu, rho0, v0)
    h0, err = integrate.quad(func_int, rho_min, rho, args=(g0_rho, Dr, eta, w10, w2_bar, nu, rho0, v0))
    return h0


if __name__ == "__main__":
    eps = 0.5
    w11 = 0.5 * eps
    w20 = 0.25 * eps ** 2
    w21 = 3 /40 + 1/2 * eps **2
    bar_w2 = 0.5 * (w20 + w21)
    Dr = 3
    eta = 3

    rho_min = 1e-6
    rho_arr = np.linspace(rho_min, 1.1879, 2000)
    varphi_prime_arr = np.zeros_like(rho_arr)
    h0_arr = np.zeros_like(rho_arr)
    g0_arr = get_g0(rho_arr, eta, Dr, w11)
    for i, rho in enumerate(rho_arr):
        varphi_prime_arr[i] = get_varphi_prime(rho, Dr, eta, w11, bar_w2)
        h0_arr[i] = get_h0(rho, Dr, eta, w11, bar_w2, rho_min=rho_min)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.plot(rho_arr, g0_arr)
    ax2.plot(rho_arr, h0_arr)
    plt.show()
    plt.close()