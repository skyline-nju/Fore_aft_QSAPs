import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_v, get_vs
from comm_tangent import get_g0


def get_varphi_prime(rho, Dr, eta, w10, w2_bar, nu=0.9, rho0=1, v0=1, rho_min=0):
    def func_int(rho, Dr, eta, w10, w2_bar, nu, rho0, v0):
        v, v1, v2 = get_vs(rho, eta, nu, rho0, v0)
        c1 = 1 - 3 * w10**2 / (2 * w2_bar)
        c2 = v / (v - 2 * Dr * w10) - 2
        return c1 * v2 / v1 + c2 * v1 / v

    int_res, err = integrate.quad(func_int, rho_min, rho, args=(Dr, eta, w10, w2_bar, nu, rho0, v0))
    varphi_prime = np.exp(-int_res)
    return varphi_prime


def get_varphi(rho, Dr, eta, w10, w2_bar, nu=0.9, rho0=1, v0=1, rho_min=0):
    varphi, err = integrate.quad(get_varphi_prime, rho_min, rho, args=(Dr, eta, w10, w2_bar, nu, rho0, v0, rho_min))
    return varphi    


def get_h0(rho, Dr, eta, w10, w2_bar, nu=0.9, rho0=1, v0=1, rho_min=0):
    def func_int(rho, Dr, eta, w10, w2_bar, nu, rho0, v0, rho_min):
        g0 = get_g0(rho, Dr, eta, w10, nu, rho0, v0)
        varphi_prime = get_varphi_prime(rho, Dr, eta, w10, w2_bar, nu, rho0, v0, rho_min)
        return g0 * varphi_prime

    G0, err = integrate.quad(func_int, rho_min, rho, args=(Dr, eta, w10, w2_bar, nu, rho0, v0, rho_min))
    varphi = get_varphi(rho, Dr, eta, w10, w2_bar, nu, rho0, v0, rho_min)
    g0 = get_g0(rho, Dr, eta, w10, nu, rho0, v0)
    return varphi * g0 - G0


def func_phase_equilibria(x_arr, Dr, eta, w10, w2_bar, nu=0.9, rho0=1, v0=1):
    """When the phase equilibira is reached, one should have [f1, f2] = [0, 0]

    Args:
        x_arr (array): [rho_g, rho_l]
        Dr (float): rotational diffusivity
        eta (float): nominal strength of the motility-density coupling
        w10 (float): strength of the fore-aft asymmetry
        w2_bar (float): (w_20 + w21) / 2
    """
    rho_g, rho_l = x_arr

    f1 = get_g0(rho_g, Dr, eta, w10, nu, rho0, v0) - get_g0(rho_l, Dr, eta, w10, nu, rho0, v0)
    
    

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
if __name__ == "__main__":
    eps = 0.5
    w11 = 0.5 * eps
    w20 = 0.125 * eps ** 2
    w21 = 3/40 + 1/4 * eps **2
    bar_w2 = 0.5 * (w20 + w21)
    Dr = 4
    eta = 3

    rho_min = 1e-6
    rho_arr = np.linspace(rho_min, 1.5, 1000)
    # rho_arr = np.logspace(-7, np.log10(1.5), 200)
    varphi_prime_arr = np.zeros_like(rho_arr)
    varphi_arr = np.zeros_like(rho_arr)
    h0_arr = np.zeros_like(rho_arr)
    g0_arr = np.zeros_like(rho_arr)

    for i, rho in enumerate(rho_arr):
        varphi_prime_arr[i] = get_varphi_prime(rho, Dr, eta, w11, bar_w2, rho_min=rho_min)
        varphi_arr[i] = get_varphi(rho, Dr, eta, w11, bar_w2, rho_min=rho_min)
        h0_arr[i] = get_h0(rho, Dr, eta, w11, bar_w2, rho_min=rho_min)
        g0_arr[i] = get_g0(rho, Dr, eta, w11)

    
    fig, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    axes[0].plot(rho_arr, varphi_arr)
    axes[1].plot(rho_arr, g0_arr)
    axes[2].plot(rho_arr, h0_arr)
    # axes[0].set_xscale("log")
    plt.show()
    plt.close()