import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import rho_f1_f2_eq
sys.path.append("..")
from scripts.read_snap import get_frame


def get_para(L):
    files = glob.glob(f"/mnt/sda/Fore_aft_QS/Offset/PD_eps_vs_Dr/L{L:g}/*.gsd")
    n = len(files)
    eps_arr = np.zeros(n)
    Dr_arr = np.zeros(n)

    for i, fi in enumerate(files):
        basename = os.path.basename(fi)
        s = basename.split("_")
        Dr_arr[i] = float(s[2].lstrip("Dr"))
        eps_arr[i] = float(s[7].lstrip("E"))
    return eps_arr, Dr_arr


def cal_var_v(L, eps_arr=None, Dr_arr=None):
    if eps_arr is None or Dr_arr is None:
        eps_arr, Dr_arr = get_para(L)
    if L == 30:
        h = 0.01
        rho0 = 80
    elif L == 20:
        h = 0.025
        rho0 = 40
    folder = f"/mnt/sda/Fore_aft_QS/Offset/PD_eps_vs_Dr/L{L:g}"
    var_v_arr = np.zeros_like(eps_arr)
    for i, eps in enumerate(eps_arr):
        Dr = Dr_arr[i]
        fname = f"{folder}/L{L}_{L}_Dr{Dr:.5f}_Dt0.000_r{rho0:g}_p{rho0:g}_e3.000_E{eps:.6f}_h{h:.3f}_1000.gsd"
        snap = get_frame(fname, i_frame=-1, flag_show=False)
        v = snap.particles.charge
        var_v_arr[i] = np.var(v)
    fout = f"/mnt/sda/Fore_aft_QS/Offset/PD_eps_vs_Dr/L{L:g}_r{rho0:g}_e3.npz"
    np.savez_compressed(fout, eps=eps_arr, Dr=Dr_arr, var_v=var_v_arr)


def plot_PD_eps_vs_Dr(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        flag_show = True
    else:
        flag_show = False
    
    fnpz = f"/mnt/sda/Fore_aft_QS/Offset/PD_eps_vs_Dr/L{L:g}_r40_e3.npz"
    with np.load(fnpz, "r") as data:
        eps, Dr, var_v = data["eps"], data["Dr"], data["var_v"]
        mask = eps < 1
        eps = eps[mask]
        Dr = Dr[mask]
        var_v = var_v[mask]
    
    eta0 = 3
    w10 = 0.5 * eps

    cb = ax.scatter(eps, Dr, c=var_v)

    qc = 1
    Dr_arr, eps_c_arr = rho_f1_f2_eq.eps_Dr_plane_3(q_max=qc)
    ax.plot(eps_c_arr, Dr_arr, c="tab:green")
    ax.fill_betweenx(Dr_arr, eps_c_arr, 1, color="tab:green", alpha=0.25)
    y = np.logspace(-3, 0, 100)
    x1 = 2 / eta0 * (1/(16 * y) + y / qc**2)
    ax.plot(x1, y, linestyle="dashed", c="tab:grey")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e-3, 1)
    ax.set_ylim(1e-3, 1.05)



    if flag_show:
        plt.show()
        plt.close()
    else:
        return cb


if __name__ == "__main__":
    L = 20
    # cal_var_v(L=20)

    plot_PD_eps_vs_Dr()