import numpy as np
import matplotlib.pyplot as plt
from read_snap import get_frame


def coarse_grain(x, dx, Lx):
    n = int(Lx / dx)
    rho = np.zeros(n)
    x += Lx/2
    for i in range(x.size):
        idx = int(x[i] / dx)
        rho[idx] += 1
    
    rho /= dx
    return rho



if __name__ == "__main__":
    folder = "/mnt/sda/Fore_aft_QS/sli/"
    seed = 211001
    dx = 1
    Lx = 60
    Dr = 4

    for phi in [40, 60, 80]:
        basename = f"L60_15_Dr{Dr:.3f}_Dt0.000_r80_p{phi:g}_e3.000_E0.500_h0.025_{seed:d}.gsd"
        fname_in = f"{folder}/binodals/{basename}"

        snap = get_frame(fname_in, i_frame=-1, flag_show=False)
        pos_x = snap.particles.position[:, 0]
        rho = coarse_grain(pos_x, dx=dx, Lx=Lx)

        x = np.linspace(dx+dx/2, Lx-dx/2, int(Lx/dx))
        plt.plot(x, rho)
    plt.show()
    plt.close()
