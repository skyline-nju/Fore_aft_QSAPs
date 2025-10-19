import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from scripts.quorum_sensing import get_eta_Pe, get_v_dv
import matplotlib.patches as mpatches


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
    return q**2 * (-w11 * eta * D2 / Pe + 0.5 * (1 + eta_prime) * D2 + 0.25 * q**2 * eta * (q**2 * w22 * Pe / 18))


def get_a3_direct(q, Pe, eta, w11, w21, w22):
    a = 4 / Pe * (0.5 * (1 + eta*(1-q**2*w21)) - w11 * eta / Pe)
    b = -1 / 36 * q**2 * w11 * eta
    c = q**2 * Pe / 72 * (1 + eta * (1 + q**2 * (w22 -w21)))
    return q**2 * (a + b + c)


def get_Routh_arr(q, Pe, eta, w11, w21, w22):
    a1 = get_a1(q, Pe, eta, w11)
    a2 = get_a2(q, Pe, eta, w11, w21)
    a3 = get_a3(q, Pe, eta, w11, w21, w22)
    # a3 = get_a3_direct(q, Pe, eta, w11, w21, w22)
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
    # Dr_arr = np.linspace(0.0001, 1, 300)
    # Dr_arr = np.logspace(-5, 1, 300)
    rho_arr = np.linspace(0, 1.7, 500)
    Dr_arr = np.linspace(0.01, 3, 300)
    # rho_arr = np.linspace(0, 1.5, 500)

    q_arr = np.logspace(-6, np.log10(q_max), 500)

    w11 = 0.5 * eps
    # w22 = 1 / 8 * eps ** 2
    # w21 = 3 /40 + 1/4 * eps **2
    w22 = w21 = 0

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
    # extent = [rho_arr.min(), rho_arr.max(), np.log10(Dr_arr.min()), np.log10(Dr_arr.max())]
    plt.imshow(state, origin="lower", extent=extent, aspect="auto")

    x, y = get_SOI_line_density_vs_Dr(eps, eta0=eta0)
    plt.plot(x, y)

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


def PeA_eta_plane(eps= 0.5, q_max=1, xlim=(-2, 2), ylim=(-4, 4)):
    w10_0 = eps * 0.5
    w22 = w21 = 0
    q_arr = np.logspace(-6, np.log10(q_max), 300)
    eta_arr = np.linspace(ylim[0], ylim[1], 1000)
    PeA_arr = np.linspace(xlim[0], xlim[1], 1000)
    PeA_arr = PeA_arr[PeA_arr != 0]
    state = np.zeros((eta_arr.size, PeA_arr.size))
    for j, eta in enumerate(eta_arr):
        for i, PeA in enumerate(PeA_arr):
            Pe = np.abs((2 * w10_0)/ PeA)
            if PeA < 0:
                w11 = -w10_0
            else:
                w11 = w10_0
            a1, Delta_2, a3 = get_Routh_arr(q_arr, Pe, eta, w11, w21, w22)
            if a3[0] < 0:
                state[j, i] = 1
            else:
                if a3.min() < 0:
                    state[j, i] = 2
                else:
                    if a1.min() < 0 or Delta_2.min() < 0:
                    # if Delta_2.min() < 0:
                    # if a1.min() < 0:
                        state[j, i] = 3

    extent = [PeA_arr[0], PeA_arr[-1], eta_arr[0], eta_arr[-1]]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(state, origin="lower", extent=extent, aspect="auto")

    y1 = eta_arr[eta_arr < 0]
    y2 = eta_arr[eta_arr > 0]
    x1 = (1 + y1) / y1
    x2 = (1 + y2) / y2
    ax.plot(x1, y1, x2, y2)

    eps = -0.6
    Dr = 0.01
    eta, Pe = get_eta_Pe(1, -0.5, Dr)
    PeA = eps / Pe
    ax.plot(PeA, eta, "o")

    Dr = 0.01
    eta, Pe = get_eta_Pe(1.1, -0.5, Dr)
    PeA = eps / Pe
    ax.plot(PeA, eta, "s")

    Dr = 1
    eta, Pe = get_eta_Pe(1, -0.5, Dr)
    PeA = eps / Pe
    ax.plot(PeA, eta, "o")

    Dr = 2
    eta, Pe = get_eta_Pe(1, -0.5, Dr)
    PeA = eps / Pe
    ax.plot(PeA, eta, "o")

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    plt.show()
    plt.close() 


def get_SOI_line(eps= 0.5, q_max=1, eta_max=6, PeA_max=1):
    w11 = eps * 0.5
    w22 = w21 = 0
    q_arr = np.logspace(-6, np.log10(q_max), 500)
    eta_arr = -np.linspace(-eta_max, 0, 5000, endpoint=False)
    PeA_arr = -np.linspace(-PeA_max, 0, 5000, endpoint=False)

    PeA_c = np.zeros_like(eta_arr)
    idx_c = 0
    for j, eta in enumerate(eta_arr):
        flag_found_SOI = False
        for i in range(idx_c, PeA_arr.size):
            Pe = 2 * w11 / PeA_arr[i]
            a1, Delta_2, a3 = get_Routh_arr(q_arr, Pe, eta, w11, w21, w22)
            if a3[0] > 0 and (Delta_2.min() < 0 or a1.min() < 0):
                PeA_c[j] = PeA_arr[i]
                idx_c = i
                flag_found_SOI = True
                break
        if not flag_found_SOI:
            break
    mask = PeA_c > 0
    return PeA_c[mask], eta_arr[mask]


def get_SOI_line_density_vs_Dr(eps=0.5, eta0=3, q_max=1, rho_max=1.7, Dr_max = 1):
    w11 = eps * 0.5
    w22 = w21 = 0
    q_arr = np.logspace(-6, np.log10(q_max), 500)
    rho_arr = np.linspace(0.1, rho_max, 500)
    Dr_arr = -np.linspace(-Dr_max, -1e-3, 500)

    Dr_c = np.zeros_like(rho_arr)
    for j, rho in enumerate(rho_arr):
        for i, Dr in enumerate(Dr_arr):
            eta, Pe = get_eta_Pe(rho, eta0, Dr)
            a1, Delta_2, a3 = get_Routh_arr(q_arr, Pe, eta, w11, w21, w22)
            if a3[0] > 0 and (Delta_2.min() < 0 or a1.min() < 0):
                Dr_c[j] = Dr
                break
    mask = Dr_c > 0
    return rho_arr[mask], Dr_c[mask]


def plot_PeA_eta_plane(ax=None, xlim=(-1, 2), ylim=(-4, 4)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        flag_show = True
    else:
        flag_show = False
    
    ymin, ymax = ylim
    xmin, xmax = xlim


    x = np.linspace(xmin, xmax, 1000)
    x1 = x [x < 1]
    x2 = x [x > 1]
    y1 = 1 / (x1-1)
    y2 = 1 / (x2 - 1)
    # ax.plot(x1, y1, c="tab:blue")
    # ax.plot(x2, y2, c="tab:blue")
    ax.fill_between(x1, y1, ymin, color="tab:blue", alpha=0.25)
    ax.fill_between(x2, y2, ymax, color="tab:blue", alpha=0.25)
    
    SOI_line = []
    alpha_list = [0.75, 0.5, 0.25]
    cm = plt.get_cmap('tab20c')
    color_list = [cm.colors[6], cm.colors[1], cm.colors[10]]
    ls = ["dotted", "--", "-"]
    for i, eps in enumerate([0.15, 0.25, 0.35]):
        x, y = get_SOI_line(eps=eps)
        line, = ax.plot(x, y, label=r"$%g$" % (eps / 2), color=color_list[i], linestyle=ls[i])
        SOI_line.append(line)
        ax.fill_betweenx(y, 0, x, color="tab:green", alpha=alpha_list[i], edgecolor="none")

    line_legend = ax.legend(handles=SOI_line, title="$|w_{1,0}|=$", loc=(0.5, 0.55),
        fontsize="large", borderpad=0.3, title_fontsize="x-large")
    ax.add_artist(line_legend)
    eta, Pe = get_eta_Pe(1, eta0=3, Dr=0.1)
    x = 0.25 / Pe
    ax.plot(x, eta, "o", fillstyle="none", color="tab:red", ms=6)
    x = 0.16 * 2 / Pe
    ax.plot(x, eta, "x", fillstyle="none", color="tab:blue",ms=6) 
    eta, Pe = get_eta_Pe(1, eta0=3, Dr=5.5)
    x = 0.25 / Pe
    ax.plot(x, eta, "p", fillstyle="none", color="tab:red", ms=6)
    x = 0.16 * 2 / Pe
    ax.plot(x, eta, "s", fillstyle="none", color="tab:blue", ms=6)    

    ax.axhline(0, color="k")
    ax.axvline(0, color="k")

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    patches = [mpatches.Patch(color="tab:blue", label = 'Long-wave stationary', alpha=0.25),
               mpatches.Patch(color="tab:green", label = 'Short-wave oscillatory', alpha=0.25)]

    ax.legend(handles=patches, loc=(0.46, 0.27), frameon=False, fontsize="large", title="Instability", title_fontsize="x-large")

    if flag_show:
        # plt.savefig("rho_linear_stab_diagram.png")
        ax.set_xlabel(r"$2D_r w_{1,0}/\bar{v}$", fontsize="xx-large")
        ax.set_ylabel(r"$\bar{\rho}\bar{v}'/\bar{v}$", fontsize="xx-large")
        # plt.show()
        plt.savefig("fig/PeA_eta.pdf")
        plt.close()


def plot_density_Dr_plane(eps, eta0=3, ax=None, xlim=(0, 1.7), ylim=(1e-3, 3)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        flag_show = True
    else:
        flag_show = False
    
    ### LSI
    rho = np.linspace(0.05, 1.5, 3000)
    v, v_prime = get_v_dv(rho, eta0)
    bar_eta = rho * v_prime / v
    Dr_LSI = v / eps * (1 + 1/bar_eta)

    # ax.plot(rho, Dr_LSI)
    ax.fill_between(rho, Dr_LSI, ylim[-1], color="tab:blue", alpha=0.25)

    x, y = get_SOI_line_density_vs_Dr(eps, eta0)
    # ax.plot(x, y)
    ax.fill_between(x, 0, y, color="tab:green", alpha=0.25)
    ax.set_xlim(xlim[0], xlim[-1])
    ax.set_ylim(ylim[0], ylim[-1])
    if flag_show:
        plt.show()
        plt.close()


if __name__ == "__main__":
    # w10_Dr_plane()
    # plot_a(1, 1e-1)

    # Dr_arr, eps_c_arr = eps_Dr_plane_3()
    # plt.loglog(eps_c_arr, Dr_arr)
    # plt.show()
    # plt.close()

    # density_Dr_plane(eps=0.25, eta0=3, q_max=1)
    PeA_eta_plane(eps=0.6)

    # for eps in [0.15, 0.25, 0.35]:
    #     PeA_c, eta = get_SOI_line(eps=eps)
    #     plt.plot(PeA_c, eta)
    # plt.show()
    # plt.close()

    # plot_PeA_eta_plane()

    # plot_density_Dr_plane(eps=0.25)
