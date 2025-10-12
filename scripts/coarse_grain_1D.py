import numpy as np
import matplotlib.pyplot as plt
from read_snap import get_frame


def coarse_grain(x, dx, Lx, Ly):
    n = int(Lx / dx)
    rho = np.zeros(n)
    x += Lx/2
    for i in range(x.size):
        idx = int(x[i] / dx)
        rho[idx] += 1
    
    rho /= (dx * Ly)
    return rho


def phase_sep_profile():
    folder = "/mnt/sda/Fore_aft_QS/sli/"
    seed = 211001
    dx = 1
    Lx = 60
    Dr = 2.3

    rho0 = 80
    for phi in [40, 60]:
        basename = f"L60_15_Dr{Dr:.3f}_Dt0.000_r80_p{phi:g}_e3.000_E0.500_h0.025_{seed:d}.gsd"
        fname_in = f"{folder}/binodals/{basename}"

        snap = get_frame(fname_in, i_frame=-1, flag_show=False)
        pos_x = snap.particles.position[:, 0]
        rho = coarse_grain(pos_x, dx=dx, Lx=Lx, Ly=15)

        x = np.linspace(dx+dx/2, Lx-dx/2, int(Lx/dx))
        plt.plot(x, rho/rho0)
    plt.show()
    plt.close()



if __name__ == "__main__":
    folder = "/mnt/sda/Fore_aft_QS/Offset/L30_r80_h0.1"

    Dr = 0.05
    rho0 = 80
    phi = 40
    Lx = 30
    Ly = 30
    h = 0.025
    seed = 1000

    dx = 1

    basename = f"L{Lx:g}_{Lx:g}_Dr{Dr:.3f}_Dt0.000_r{rho0:g}_p{phi:g}_e3.000_E0.500_h{h:.3f}_{seed:d}.gsd"
    fname_in = f"{folder}/{basename}"

    snap = get_frame(fname_in, i_frame=20, flag_show=False)
    pos_x = snap.particles.position[:, 0]
    rho = coarse_grain(pos_x, dx=dx, Lx=Lx, Ly=Ly)

    x = np.linspace(dx+dx/2, Lx-dx/2, int(Lx/dx))
    plt.plot(x, rho)
    plt.axhline(30)
    plt.show()
    plt.close()
