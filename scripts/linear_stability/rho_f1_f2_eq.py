import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_eta_Pe


def get_eta_prime(q, eta, w21):
    return eta * (1 - q**2 * w21)

def get_D2(q, Pe):
    return 4 / Pe + q**2 * Pe / 36

def get_a1(q, Pe, eta, w11):
    return 5 / Pe + q**2 * (Pe / 36 - w11 * eta)


def get_a2(q, Pe, eta, w11, w21):
    return 4 / Pe**2 + q**2 * (7/9 - 5 * w11 * eta / Pe + eta / 2) - q**4 * eta * (w11 * Pe / 36 + w21 / 2)


def get_a3(q, Pe, eta, w11, w21, w22):
    eta_prime = get_eta_prime(q, eta, w21)
    D2 = get_D2(q, Pe)
    return q**2 * (-w11 * eta * D2 / Pe + 0.5 * (1 + eta_prime) * D2 + 0.25 * q**2 * eta * (q**2 * w22 * Pe / 18 + w11))


def get_a3_direct(q, Pe, eta, w11, w21, w22):
    a = 4 / Pe * (0.5 * (1 + eta*(1-q**2*w21)) - w11 * eta / Pe)
    b = 2 / 9 * q**2 * w11 * eta
    c = q**2 * Pe / 72 * (1 + eta * (1 + q**2 * (w22 -w21)))
    return q**2 * (a + b + c)


def get_Routh_arr(q, Pe, eta, w11, w21, w22):
    a1 = get_a1(q, Pe, eta, w11)
    a2 = get_a2(q, Pe, eta, w11, w21)
    a3 = get_a3(q, Pe, eta, w11, w21, w22)
    Delta_2 = a1 * a2 - a3
    return a1, Delta_2, a3


def plot_a(eps, Dr, eta0=3):
    # q_arr = np.linspace(0, 2, 100)
    q_arr = np.logspace(-5, 0, 1000)

    rho = 1

    eta, Pe = get_eta_Pe(rho, eta0, Dr)
    w11 = 0.5 * eps
    w22 = 1 / 8 * eps ** 2
    w21 = 3 /40 + 1/4 * eps **2

    a1 = get_a1(q_arr, Pe, eta, w11)
    a2 = get_a2(q_arr, Pe, eta, w11, w21)
    eta_prime = get_eta_prime(q_arr, eta, w21)
    D2 = get_D2(q_arr, Pe)
    y = q_arr**2 / 4 - q_arr**2 * w11 * eta * (1/Pe + D2) + D2/Pe + 0.5 * q_arr**2 * (1+eta_prime)

    a3 = get_a3(q_arr, Pe, eta, w11, w21, w22)
    a33 = get_a3_direct(q_arr, Pe, eta, w11, w21, w22)

    fig, axes = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=(8, 8))

    axes[0].plot(q_arr, a1)
    axes[1].plot(q_arr, a2)
    axes[2].plot(q_arr, a3)
    axes[3].plot(q_arr, a1 * a2 - a3)

    axes[3].set_xlabel(r"$q$", fontsize="xx-large")
    axes[0].set_ylabel(r"$a_1$", fontsize="xx-large")
    axes[1].set_ylabel(r"$a_2$", fontsize="xx-large")
    axes[2].set_ylabel(r"$a_3$", fontsize="xx-large")
    axes[3].set_ylabel(r"$a_1 a_2 - a_3$", fontsize="xx-large")

    for ax in axes:
        ax.axhline(0, linestyle="dashed")
        ax.set_xscale("log")
    plt.show()
    plt.close()


def density_Dr_plane(eps=0.5, eta0=3, q_max=1):
    rho_arr = np.linspace(0, 1.5, 500)
    Dr_arr = np.linspace(0.01, 5, 300)

    q_arr = np.logspace(-6, np.log10(q_max), 500)

    w11 = 0.5 * eps
    w22 = 1 / 8 * eps ** 2
    w21 = 3 /40 + 1/4 * eps **2

    state = np.zeros((Dr_arr.size, rho_arr.size))
    for j, Dr in enumerate(Dr_arr):
        for i, rho in enumerate(rho_arr):
            eta, Pe = get_eta_Pe(rho, eta0, Dr)
            a1, Delta_2, a3 = get_Routh_arr(q_arr, Pe, eta, w11, w21, w22)
            if a3[0] < 0:
                state[j, i] = 1
            else:
                if a3.min() < 0:
                    state[j, i] = 2
                else:
                    if a1.min() < 0 or Delta_2.min() < 0:
                        state[j, i] = 3
    
    extent = [rho_arr.min(), rho_arr.max(), Dr_arr.min(), Dr_arr.max()]
    plt.imshow(state, origin="lower", extent=extent, aspect="auto")

    plt.axvline(0.5, linestyle="dashed", color="w")
    plt.axvline(1, linestyle="dashed", color="w")
    plt.show()
    plt.close()


def eps_Dr_plane_3(eta0=3, flag_show=False, q_max=1):
    eps_min = 8e-3
    eps_max = 1e0
    eps_arr = np.logspace(np.log10(eps_min), np.log10(eps_max), 500)

    Dr_min = 9e-4
    Dr_max = 1
    Dr_arr = np.logspace(np.log10(Dr_min), np.log10(Dr_max), 500)

    q_arr = np.logspace(-6, np.log10(q_max), 500)
    rho=1

    state = np.zeros((Dr_arr.size, eps_arr.size))
    for j, Dr in enumerate(Dr_arr):
        eta, Pe = get_eta_Pe(rho, eta0, Dr)
        for i, eps in enumerate(eps_arr):
            w11 = 0.5 * eps
            # w22 = 1 / 8 * eps ** 2
            # w21 = 3 /40 + 1/4 * eps **2
            w22 = w21 = 0
            a1, Delta_2, a3 = get_Routh_arr(q_arr, Pe, eta, w11, w21, w22)
            state[j, i] = Delta_2.min()

    if flag_show:
        state[state <0] = -1
        state[state >=0] = 1
        extent=[np.log10(eps_min), np.log10(eps_max), np.log10(Dr_min), np.log10(Dr_max)]
        plt.imshow(state, origin="lower", extent=extent)
        plt.show()
        plt.close()
    else:
        eps_c = np.zeros_like(Dr_arr)
        for j, Dr in enumerate(Dr_arr):
            for i, eps in enumerate(eps_arr[:-1]):
                if state[j, i] >= 0 and state[j, i+1] < 0:
                    eps_c[j] = eps
        return Dr_arr, eps_c


if __name__ == "__main__":
    # w10_Dr_plane()
    # plot_a(1, 1e-1)

    # Dr_arr, eps_c_arr = eps_Dr_plane_3()
    # plt.loglog(eps_c_arr, Dr_arr)
    # plt.show()
    # plt.close()

    density_Dr_plane(eps=0.25, eta0=3, q_max=1)