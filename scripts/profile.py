import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from scripts.read_snap import get_frame, get_nframe


epyc01 = "/run/user/1000/gvfs/sftp:host=10.10.9.150,user=ps/home/ps/data"
epyc02 = "/run/user/1000/gvfs/sftp:host=10.10.9.158,user=ps/home/ps/data"
sli = "/run/user/1000/gvfs/sftp:host=10.10.14.155,port=10066,user=sli/home/sli/yduan"


def coarse_grain(x, dx, Lx, Ly):
    n = int(Lx / dx)
    rho = np.zeros(n)
    if x.min() < 0:
        x += Lx / 2
    for i in range(x.size):
        idx = int(x[i] / dx)
        rho[idx] += 1
    rho /= (dx * Ly)
    return rho


def coarse_grain_rho_p_v(x, theta, v, dx, Lx, Ly):
    n = int(Lx / dx)
    rho = np.zeros(n)
    px = np.zeros(n)
    vel = np.zeros(n)
    if x.min() < 0:
        x += Lx / 2
    for i in range(x.size):
        idx = int(x[i] / dx)
        rho[idx] += 1
        px[idx] += np.cos(theta[i])
        vel[idx] += v[i]
    rho /= (dx * Ly)
    px /= (dx * Ly)
    vel /= (dx * Ly)
    return rho, px, vel


def shift_to_center(x, Lx):
    if x.min() < 0:
        x += Lx / 2
    
    theta = x / Lx * 2 * np.pi
    theta_m = np.atan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    if theta_m < 0:
        theta_m += np.pi * 2
    xm = theta_m / (2 * np.pi) * Lx
    x_new = x - (xm - Lx/2)
    x_new[x_new < 0] += Lx
    x_new[x_new >= Lx] -= Lx
    return x_new


def get_time_ave_profiles(fname, Lx, Ly, dx=0.25, beg_frame=30, flag_only_rho=True, load_existed_data=False):
    basename = os.path.basename(fname)
    fout = f"tmp/profiles/{basename.replace(".gsd", ".npz")}"
    if load_existed_data and os.path.exists(fout):
        with np.load(fout, "r") as data:
            x = data["x"]
            rho_x = data["rho"]
            print("load profiles from", fout)
            if flag_only_rho:
                return x, rho_x
            else:
                p_x = data["p"]
                v_x = data["v"]
                return x, rho_x, p_x, v_x
    nx = int(Lx / dx)
    rho_x = np.zeros(nx)
    if not flag_only_rho:
        p_x = np.zeros(nx)
        v_x = np.zeros(nx)
    x = np.linspace(dx+dx/2, Lx-dx/2, nx)

    n_frame = get_nframe(fname)
    for i in range(beg_frame, n_frame):
        snap = get_frame(fname, i_frame=i, flag_show=False)
        pos_x = snap.particles.position[:, 0]
        pos_x = shift_to_center(pos_x, Lx)
        if not flag_only_rho:
            theta = snap.particles.position[:, 2]
            v = snap.particles.charge
        if flag_only_rho:
            rho_x += coarse_grain(pos_x, dx=dx, Lx=Lx, Ly=Ly)
        else:
            rho_i, p_i, v_i = coarse_grain_rho_p_v(pos_x, theta, v, dx, Lx, Ly)
            rho_x += rho_i
            p_x += p_i
            v_x += v_i
    rho_x /= (n_frame - beg_frame)
    if not flag_only_rho:
        p_x /= (n_frame - beg_frame)
        v_x /= (n_frame - beg_frame)
        np.savez_compressed(fout, x=x, rho=rho_x, p=p_x, v=v_x)
        return x, rho_x, p_x, v_x
    else:
        np.savez_compressed(fout, x=x, rho=rho_x)
        return x, rho_x


def plot_instant_profile(fname_in, Lx, dx=0.25, rho0=80, i_frame=-1, ax=None):
    if ax is None:
        ax = plt.subplots(1, 1, constrained_layout=True)
        flag_show = True
    else:
        flag_show = False
    snap = get_frame(fname_in, i_frame=i_frame, flag_show=False)
    pos_x = snap.particles.position[:, 0]
    pos_x = shift_to_center(pos_x, Lx)
    rho = coarse_grain(pos_x, dx=dx, Lx=Lx, Ly=15)

    x = np.linspace(dx+dx/2, Lx-dx/2, int(Lx/dx))
    ax.plot(x, rho/rho0)

    if flag_show:
        plt.show()
        plt.close()


def plot_averaged_rho():
    folder = "/mnt/sda/Fore_aft_QS/sli/"

    rho0 = 80
    Dr = 3
    for phi in [40, 60, 80]:
        basename = f"L60_15_Dr{Dr:.3f}_Dt0.000_r{rho0:g}_p{phi:g}_e3.000_E0.500_h0.025_211001.gsd"
        fname_in = f"{folder}/binodals/{basename}"
        if phi == 20:
            beg_frame = 170
        else:
            beg_frame = 80
        x, rho_x = get_time_ave_profiles(fname_in, Lx=60, Ly=15, dx=0.25, beg_frame=beg_frame)
        plt.plot(x, rho_x/rho0)


        # if phi == 80:
        #     mask = np.logical_and(x > 10, x < 50)
        #     rho_l = np.mean(rho_x[mask]) / rho0
        #     print("rho_l=", rho_l)
        #     plt.axhline(rho_l, linestyle="dashed")

    # plt.yscale("log")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # folder = f"{epyc01}/Fore_aft_QS/Offset2/L60_20_r80_e0.25"
    folder = f"{epyc01}/Fore_aft_QS/Offset2/L50_12.5_r80"

    Lx = 50
    Ly = 12.5
    eta = 3
    seed = 2000
    rho0 = 80
    Dr = 1.8
    eps = 0.25
    beg_frame = 150

    phi = 20

    basename = f"L{Lx:g}_{Ly:g}_Dr{Dr:.3f}_Dt0.000_r{rho0:g}_p{phi:g}_e{eta:.3f}_E{eps:.3f}_h0.025_{seed:d}.gsd"
    fname_in = f"{folder}/{basename}"


    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    dx = 1
    plot_instant_profile(fname_in, Lx, dx=dx, rho0=80, i_frame=-100, ax=ax)
    plot_instant_profile(fname_in, Lx, dx=dx, rho0=80, i_frame=-50, ax=ax)
    plot_instant_profile(fname_in, Lx, dx=dx, rho0=80, i_frame=-1, ax=ax)
    plt.show()
    plt.close()

    dx = 0.25
    fig, axes = plt.subplots(1, 1, sharex=True, constrained_layout=True)
    for phi in [20, 24]:
        basename = f"L{Lx:g}_{Ly:g}_Dr{Dr:.3f}_Dt0.000_r{rho0:g}_p{phi:g}_e{eta:.3f}_E{eps:.3f}_h0.025_{seed:d}.gsd"
        fname_in = f"{folder}/{basename}"
        x, rho_x = get_time_ave_profiles(fname_in, Lx=Lx, Ly=Ly, dx=dx, beg_frame=beg_frame, flag_only_rho=True)
        axes.plot(x/Lx, rho_x/rho0)

    # x, rho_x, p_x, v_x = get_time_ave_profiles(fname_in, Lx=Lx, Ly=Ly, dx=0.25, beg_frame=beg_frame, flag_only_rho=False)
    # axes[0].plot(x/Lx, rho_x/rho0)
    # vm = v_x / rho_x
    # axes[1].plot(x/Lx, np.log(v_x) + Dr * eps / vm)
    # Lx = 60
    # folder = f"{epyc01}/Fore_aft_QS/Offset2/L{Lx}_20_r80_e0.25"
    # basename = f"L{Lx:g}_{Ly:g}_Dr{Dr:.3f}_Dt0.000_r{rho0:g}_p{phi:g}_e{eta:.3f}_E{eps:.3f}_h0.025_{seed:d}.gsd"
    # fname_in = f"{folder}/{basename}"

    # x, rho_x = get_time_ave_profiles(fname_in, Lx=Lx, Ly=Ly, dx=0.25, beg_frame=beg_frame, flag_only_rho=True)
    # axes.plot(x/Lx, rho_x/rho0)

    plt.show()
    plt.close()