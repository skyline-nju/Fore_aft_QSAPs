import numpy as np
import matplotlib.pyplot as plt
from read_snap import get_frame, get_nframe


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


def time_ave(fname, Lx, Ly, dx=0.25, beg_frame=30):
    nx = int(Lx / 0.25)
    rho_x = np.zeros(nx)
    x = np.linspace(dx+dx/2, Lx-dx/2, nx)

    n_frame = get_nframe(fname)
    for i in range(beg_frame, n_frame):
        snap = get_frame(fname, i_frame=i, flag_show=False)
        pos_x = snap.particles.position[:, 0]
        pos_x = shift_to_center(pos_x, Lx)
        rho_x += coarse_grain(pos_x, dx=dx, Lx=Lx, Ly=Ly)
    rho_x /= (n_frame - beg_frame)
    return x, rho_x


def plot_instant_profile():
    folder = "/mnt/sda/Fore_aft_QS/sli/"
    seed = 211001
    dx = 0.25
    Lx = 60
    Dr = 3

    rho0 = 80
    for phi in [40, 60, 80]:
        basename = f"L60_15_Dr{Dr:.3f}_Dt0.000_r80_p{phi:g}_e3.000_E0.500_h0.025_{seed:d}.gsd"
        fname_in = f"{folder}/binodals/{basename}"

        snap = get_frame(fname_in, i_frame=-2, flag_show=False)
        pos_x = snap.particles.position[:, 0]
        pos_x = shift_to_center(pos_x, Lx)
        rho = coarse_grain(pos_x, dx=dx, Lx=Lx, Ly=15)

        x = np.linspace(dx+dx/2, Lx-dx/2, int(Lx/dx))
        plt.plot(x, rho/rho0)
    plt.show()
    plt.close()



if __name__ == "__main__":
    # plot_instant_profile()
    folder = "/mnt/sda/Fore_aft_QS/sli/"

    rho0 = 80
    Dr = 2.3
    for phi in [40, 60]:
        basename = f"L60_15_Dr{Dr:.3f}_Dt0.000_r{rho0:g}_p{phi:g}_e3.000_E0.500_h0.025_211001.gsd"
        fname_in = f"{folder}/binodals/{basename}"
        x, rho_x = time_ave(fname_in, Lx=60, Ly=15, dx=0.25, beg_frame=30)
        plt.plot(x, rho_x/rho0)

    # plt.yscale("log")
    plt.show()
    plt.close()