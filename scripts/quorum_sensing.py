import numpy as np
import matplotlib.pyplot as plt


def get_v(rho, eta, nu=0.9, rho0=1, v0=1):
    return v0 * (1 + nu * np.tanh(eta/nu * (rho-rho0)/rho0))


def get_v_deriv1(rho, eta, nu=0.9, rho0=1, v0=1):
    v = get_v(rho, eta, nu, rho0, v0)
    return v0 * eta / rho0 * (1 - ((v/v0 - 1)/nu) ** 2)


def get_v_deriv2(rho, eta, nu=0.9, rho0=1, v0=1):
    v = get_v(rho, eta, nu, rho0, v0)
    v1 = get_v_deriv1(rho, eta, nu, rho0, v0)
    return -2 * eta * v1 /(rho0 * nu**2) * (v/v0 - 1)


def get_v_dv(rho, eta, nu=0.9, rho0=1, v0=1):
    v = get_v(rho, eta, nu, rho0, v0)
    dv = v0 * eta / rho0 * (1 - ((v/v0 - 1)/nu) ** 2)
    return v, dv


def get_vs(rho, eta, nu=0.9, rho0=1, v0=1):
    v = get_v(rho, eta, nu, rho0, v0)
    dv = v0 * eta / rho0 * (1 - ((v/v0 - 1)/nu) ** 2)
    ddv = -2 * eta * dv /(rho0 * nu**2) * (v/v0 - 1)
    return v, dv, ddv


def get_rho(v, eta, nu=0.9, rho0=1, v0=1):
    return nu / eta * np.arctanh((v/v0 - 1)/nu) * rho0 + rho0


def plot_v_dv_ddv():
    eps = 0.5
    w10 = 0.5 * eps

    nu = 0.9

    eta = 3

    rho_arr = np.linspace(0, 2, 500)
    v_arr, dv_arr, ddv_arr = get_vs(rho_arr, eta, nu)

    fig, axes = plt.subplots(3, 1, figsize=(8,8), sharex=True, constrained_layout=True)
    axes[0].plot(rho_arr, v_arr)
    axes[1].plot(rho_arr, dv_arr)
    axes[2].plot(rho_arr, ddv_arr)
    axes[0].axhline(0.1, linestyle=":")
    axes[0].axhline(1.9, linestyle=":")
    axes[0].axvline(1, linestyle="--")
    axes[1].axvline(1, linestyle="--")
    axes[2].axvline(1, linestyle="--")
    axes[2].set_xlabel(r"$\bar{\rho}/\rho_0$", fontsize="xx-large")
    axes[0].set_ylabel(r"$v$", fontsize="xx-large")
    axes[1].set_ylabel(r"$v'$", fontsize="xx-large")
    axes[2].set_ylabel(r"$v''$", fontsize="xx-large")
    plt.show()
    plt.close()


if __name__ == "__main__":
    rho_arr = np.linspace(0, 2, 200)

    eta0 = 3
    v, v_prime = get_v_dv(rho_arr, eta=eta0)

    eta = rho_arr * v_prime / v

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.plot(rho_arr, v)
    ax2.plot(rho_arr, eta)
    plt.show()
    plt.close()
