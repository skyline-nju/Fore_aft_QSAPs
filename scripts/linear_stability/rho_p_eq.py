import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_eta_Pe


def cal_v_v_prime(rho, eta, rho0=1, v0=1, kappa=0.9):
    f_tanh = np.tanh(eta/kappa * (rho-rho0)/rho0)
    v = v0 * (1 + kappa * f_tanh)
    v_prime = v0 * eta / rho0 * (1 - f_tanh**2)
    return v, v_prime


def eps_eta_plane(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 6))
        flag_show = True
    else:
        flag_show = False
    
    ymin, ymax = -5, 5
    xmin, xmax = -4, 4

    x = np.linspace(xmin, xmax, 1000)
    x1 = x [x < 1]
    x2 = x [x > 1]
    y1 = 1 / (x1-1)
    y2 = 1 / (x2 - 1)
    ax.plot(x1, y1, c="tab:blue")
    ax.plot(x2, y2, c="tab:blue")
    ax.fill_between(x1, y1, ymin, color="tab:blue", alpha=0.25)
    ax.fill_between(x2, y2, ymax, color="tab:blue", alpha=0.25)
    
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")

    # Short-wave Instability arising from the coupled rho-p equations
    # w_{l, m} for the isotropic kernel shifted by distance eps
    eps = 0.5
    w11 = 0.5 * eps
    w20 = 1 / 8 * eps ** 2
    w21 = 3 /40 + 1/4 * eps **2
    bar_w2 = 0.5 * (w20 + w21)

    # Tr(M) = 0
    qc = 1
    # y = np.linspace(1/(2 * w11), ymax, 10000)
    y = np.linspace(1, ymax, 1000)
    x = y * w11**2 * qc**2 + 0.5 * qc * w11 * np.sqrt(4 * y**2 * w11**2 * qc**2 -1)
    ax.plot(x, y, c="tab:orange")
    ax.fill_betweenx(y, 0, x, color="tab:orange", alpha=0.25)
    
    # Det(M) = 0
    y = np.linspace(1e-3, ymax, 1000)
    x_l = 1 - 2 * bar_w2 + 1 / y
    ax.plot(x_l, y, c="tab:green")

    x_r = 1/y + 1
    ax.fill_betweenx(y, x_l, x_r, color="tab:green", alpha=0.25) 

    phi = 80 / 80
    eta, Pe = get_eta_Pe(phi, eta0=3, Dr=0.1)
    x = 0.25 / Pe
    ax.plot(x, eta, "o")

    x = 0.16 * 2 / Pe
    ax.plot(x, eta, "o")

    eta, Pe = get_eta_Pe(phi, eta0=3, Dr=5.5)
    x = 0.25 / Pe
    ax.plot(x, eta, "o")
    x = 0.16 * 2 / Pe
    ax.plot(x, eta, "o")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    patches = [mpatches.Patch(color="tab:blue", label = 'Long-wave Stationary', alpha=0.25),
               mpatches.Patch(color="tab:green", label = 'Short-wave Stationary', alpha=0.25),
               mpatches.Patch(color="tab:orange", label = 'Short-wave Oscillatory', alpha=0.25)
               ]

    ax.legend(handles=patches, loc="upper left", frameon=False, fontsize="x-large", title="Instability", title_fontsize="x-large")
    ax.set_xlabel(r"$2D_r w_{1,1}/\bar{v}$", fontsize="xx-large")
    ax.set_ylabel(r"$\bar{\rho}\bar{v}'/\bar{v}$", fontsize="xx-large")
    if flag_show:
        plt.show()
        # plt.savefig("linPD_eps_eta.pdf")
        plt.close()



def density_Dr_plane():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    qc = 1.
    eps = 0.1
    w11 = 0.5 * eps
    w20 = 1 / 8 * eps ** 2
    w21 = 3 /40 + 1/4 * eps **2
    bar_w2 = 0.5 * (w20 + w21)

    eta = 3
    rho = np.linspace(0.05, 1.5, 3000)
    v, v_prime = cal_v_v_prime(rho, eta)
    bar_eta = rho * v_prime / v
    Dr_LSI = v / (2 * w11) * (1 + 1/bar_eta)
    Dr_SSI = v / (2 * w11) * (1 + 1/bar_eta - 2 * qc**2 * bar_w2)

    Dr_zero = v / (2 * w11)
    
    ax.plot(rho, Dr_LSI)
    ax.plot(rho, Dr_SSI)
    ax.plot(rho, Dr_zero, "--")

    #SOI
    mask = w11 * bar_eta > 0.5 / qc
    if mask.size > 0:
        rho_mask = rho[mask]
        Dr_SOI = 0.5 * qc**2 * v[mask] * bar_eta[mask] * (w11 + np.sqrt(w11**2 - 0.25 / (qc * bar_eta[mask])**2))

        ax.plot(rho_mask, Dr_SOI)

    ax.set_ylim(0, 5)
    ax.set_xlim(rho.min(), rho.max())
    # ax.axhline(2.3)
    plt.show()
    plt.close()


if __name__ == "__main__":
    eps_eta_plane()
    # density_Dr_plane()