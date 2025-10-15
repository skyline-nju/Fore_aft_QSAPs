import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def LSI_PeA_vs_eta(ax=None, xlim=(-4, 4), ylim=(-5, 5)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 6))
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
    ax.plot(x1, y1, c="tab:blue")
    ax.plot(x2, y2, c="tab:blue")
    ax.fill_between(x1, y1, ymin, color="tab:blue", alpha=0.25)
    ax.fill_between(x2, y2, ymax, color="tab:blue", alpha=0.25)
    
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    patches = [mpatches.Patch(color="tab:blue", label = 'Long-wave Stationary\nInstability (LSI)', alpha=0.25)]

    ax.legend(handles=patches, loc="upper left", frameon=False, fontsize="large")
    ax.set_xlabel(r"$2D_r w_{1,1}/\bar{v}$", fontsize="xx-large")
    ax.set_ylabel(r"$\bar{\rho}\bar{v}'/\bar{v}$", fontsize="xx-large")
    if flag_show:
        # plt.savefig("rho_linear_stab_diagram.png")
        plt.show()
        plt.close()
    


if __name__ == "__main__":
    LSI_PeA_vs_eta()
