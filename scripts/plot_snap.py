import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from read_snap import get_frame
from gsd import hoomd
import sys


def add_xlabel(fig, axes, loc, para_name, para_arr, fontsize):
    if loc == "bottom":
        j = axes.shape[0] - 1
        va = "top"
    elif loc == "top":
        j = 0
        va = "bottom"
    else:
        print("Error, loc must be 'top' or 'bottom'")
        sys.exit(1)

    for i, para in enumerate(para_arr):
        if i == 0:
            if para_name == "replica":
                text = r"replica $%d$" % (i)
            else:
                text = r"$%g$" % (para)
        else:
            if para_name == "replica":
                text = r"%d" % i
            else:
                text = r"$%g$" % para
        # bbox = [xmin, ymin, xmax, ymax]
        bbox = axes[j, i].get_position().get_points().flatten()
        x = (bbox[0] + bbox[2]) * 0.5
        if loc == "top":
            y = bbox[3]
        else:
            y = bbox[1]
        fig.text(x, y, text, fontsize=fontsize, ha="center", va=va)


def add_ylabel(fig, axes, loc, para_name, para_arr, fontsize, vertical=True, reverse=False):
    if loc == "left":
        col = 0
        ha = "right"
    elif loc == "right":
        col = axes.shape[1] - 1
        ha = "left"
    else:
        print("Error, loc must be 'left' or 'right'")
        sys.exit(1)
    if vertical:
        rot = "vertical"
    else:
        rot = "horizontal"

    for j, para in enumerate(para_arr):
        if para_name == "replica":
            text = r"$%g$" % j
        elif para_name == "":
            text = r"$%g$" % para
        else:
            text = "$%g$" % para

            # if j == 0:
            #     text = r"$%s=%g$" % (para_name, para)
            # else:
            #     text = "$%g$" % para
        
        if reverse:
            row = axes.shape[0] - 1 - j
        else:
            row = j
        
        bbox = axes[row, col].get_position().get_points().flatten()
        if loc == "left":
            x = bbox[0]
        else:
            x = bbox[2]
        y = (bbox[1] + bbox[3]) * 0.5
        fig.text(x, y, text, fontsize=fontsize, ha=ha, va="center",
            rotation=rot)


def plot_one_panel(fname, i_frame=-1, ax=None, ms=0.04, color_coding="none", frac=1.):
    snap = get_frame(fname, i_frame)
    if snap is not None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            flag_show = True
        else:
            flag_show = False
        
        Lx = snap.configuration.box[0]
        Ly = snap.configuration.box[1]
        x = snap.particles.position[:, 0] + Lx / 2
        y = snap.particles.position[:, 1] + Ly / 2
        theta = snap.particles.position[:,2]
        theta[theta <0] += np.pi * 2
        if frac < 1.:
            n = int(x.size * frac)
            x, y, theta = x[:n], y[:n], theta[:n]
        if color_coding == "none":
            ax.plot(x, y, ".", ms=ms, c="b", alpha=1)
        elif color_coding == "ori":
            # ax.scatter(x, y, s=ms, c=theta, cmap="hsv", alpha=1, edgecolors="none")
            ax.scatter(x, y, s=ms, c=theta, cmap="hsv", alpha=0.8)

        ax.axis("scaled")
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        if flag_show:
            plt.show()
            plt.close()
        

def PD_rho_Dr(eps, ms=0.05, frac=1., eta0=3):
    if eps == 0.5:
        L = 30
        rho0 = 80
        h = 0.025
        seed = 1000
        root = "/mnt/sda/Fore_aft_QS/Offset/L30_r80"
        rho_arr = np.array([10, 30, 50, 70, 90, 110])
        Dr_arr = np.array([0.1, 0.3, 0.7, 1, 1.5, 2, 3])
    nrows = Dr_arr.size
    ncols = rho_arr.size
    if ncols == nrows:
        figsize = (ncols * 2 + 0.25, nrows * 2 + 0.25)
    elif ncols > nrows:
        figsize = (ncols * 2 + 0.25, nrows * 2 + 0.5)
    else:
        figsize = (ncols * 2 + 0.5, nrows * 2 + 0.25)
    print(figsize)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)


    file_pat = f"{root}/L{L:d}_{L:d}_Dr%.3f_Dt0.000_r{rho0:g}_p%g_e{eta0:.3f}_E{eps:.3f}_h{h:.3f}_{seed}.gsd"

    # rect = [-0.01, -0.01, 1.01, 1.01]
    rect = [0.02, 0.015, 1.005, 1.01]
    for j, Dr in enumerate(Dr_arr[::-1]):
        for i, rho in enumerate(rho_arr):
            fin = file_pat % (Dr, rho)
            print(fin)
            if os.path.exists(fin):
                plot_one_panel(fin, ax=axes[j, i], ms=ms, color_coding="none", frac=frac)
                # print("find", fin)
            axes[j, i].axis("off")
    plt.tight_layout(h_pad=0.1, w_pad=0.1, pad=1.1, rect=rect)
    fontsize = 30
    add_xlabel(fig, axes, "bottom", "", rho_arr / 80, fontsize)
    add_ylabel(fig, axes, "left", "", Dr_arr, fontsize, vertical=True, reverse=True)

    fout_name = f"fig/snaps.jpg"

    plt.savefig(fout_name, dpi=300)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    root = "/mnt/sda/Fore_aft_QS/Offset/L30_r80"
    
    fname = f"{root}/L30_30_Dr2.500_Dt0.000_r80_p60_e3.000_E0.500_h0.025_1000.gsd"
    # plot_one_panel(fname)

    PD_rho_Dr(eps=0.5, frac=0.5, ms=0.05)

    # Dr_arr = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
    # gamma_arr = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20])

    # PD_gamma_Dr(gamma_arr, Dr_arr, ms=0.025, frac=0.5, eta=4)
    # PD_gamma_Dr(gamma_arr, Dr_arr, ms=0.025, frac=0.5, eta=2)