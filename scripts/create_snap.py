import sys
import numpy as np
import matplotlib.pyplot as plt
from gsd import hoomd
from read_snap import get_frame
import os
import sys


def scale(s: hoomd.Frame, nx: int, ny: int, eps=0) -> hoomd.Frame:
    lx = s.configuration.box[0]
    ly = s.configuration.box[1]
    Lx, Ly = lx * nx, ly * ny

    N = s.particles.N * nx * ny
    pos = np.zeros((N, 3), dtype=np.float32)
    # type_id = np.zeros(N, dtype=np.uint32)
    charge = np.zeros(N, dtype=np.float32)
    for i in range(nx * ny):
        beg = i * s.particles.N
        end = beg + s.particles.N
        pos[beg:end, 0] = s.particles.position[:, 0] * nx
        pos[beg:end, 1] = s.particles.position[:, 1] * ny
        pos[beg:end, 2] = s.particles.position[:, 2]
        # type_id[beg:end] = s.particles.typeid
        charge[beg:end] = s.particles.charge
    if nx > 1:
        pos[:, 0] += (np.random.rand(N) - 0.5) * eps * nx
        mask = pos[:, 0] < Lx/2
        pos[:, 0][mask] += Lx
        mask = pos[:, 0] >= Lx/2
        pos[:, 0][mask] -= Lx
    if ny > 1:
        pos[:, 1] += (np.random.rand(N) - 0.5) * eps * ny
        mask = pos[:, 1] < Ly/2
        pos[:, 1][mask] += Ly
        mask = pos[:, 1] >= Ly/2
        pos[:, 1][mask] -= Ly
    s2 = hoomd.Frame()
    s2.configuration.box = [Lx, Ly, 1, 0, 0, 0]
    s2.particles.N = N
    s2.particles.position = pos
    # s2.particles.typeid = type_id
    s2.particles.charge = charge
    # s2.particles.types = s.particles.types
    s2.configuration.step = 0
    return s2


def adjust_density(s: hoomd.Frame, rho_new:float) -> hoomd.Frame:
    Lx = s.configuration.box[0]
    Ly = s.configuration.box[1]
    N = s.particles.N
    N_new = int(Lx * Ly * rho_new)
    pos = np.zeros((N_new, 3), dtype=np.float32)
    charge = np.zeros(N_new, dtype=np.float32)

    rng = np.random.default_rng()
    data = np.zeros((N, 4), dtype=np.float32)
    data[:, 0:3] = s.particles.position
    data[:, 3] = s.particles.charge
    rng.shuffle(data, axis=0)
    if N_new > N:
        pos[:N] = data[:, 0:3]
        charge[:N] = data[:, 3]
        pos[N:N_new] = data[N_new-N, 0:3]
        charge[N:N_new] = data[N_new-N, 3]
    else:
        pos[:N_new] = data[:N_new, 0:3]
        charge[:N_new] = data[:N_new, 3]

    s2 = hoomd.Frame()
    s2.configuration.box = [Lx, Ly, 1, 0, 0, 0]
    s2.particles.N = N_new
    s2.particles.position = pos
    s2.particles.charge = charge
    s2.configuration.step = 0
    return s2


if __name__ == "__main__":
    folder = "/mnt/sda/Fore_aft_QS/sli/"
    seed = 211001
    Dr = 5
    basename = f"L60_15_Dr{Dr:.3f}_Dt0.000_r80_p60_e3.000_E0.500_h0.025_{seed:d}.gsd"
    fname_in = f"{folder}/binodals/{basename}"

    snap = get_frame(fname_in, flag_show=True)

    ### Just change filename
    # Dr = 5
    # fname_out = f"{folder}/built_snap/L60_15_Dr{Dr:.3f}_Dt0.000_r80_p40_e3.000_E0.500_h0.025_{seed:d}.gsd"
    # with hoomd.open(name=fname_out, mode="w") as fout:
    #     fout.append(snap)

    ### Scale up
    # Dr = 5
    # snap2 = scale(snap, 2, 1)
    # fname_out = f"{folder}/built_snap/L60_15_Dr{Dr:.3f}_Dt0.000_r80_p40_e3.000_E0.500_h0.025_21{seed:d}.gsd"
    # with hoomd.open(name=fname_out, mode="w") as fout:
    #     fout.append(snap2)


    ### Adjust the density
    phi = 80
    snap2 = adjust_density(snap, phi)
    fname_out = f"{folder}/built_snap/L60_15_Dr{Dr:.3f}_Dt0.000_r80_p{phi:g}_e3.000_E0.500_h0.025_{seed:d}.gsd"
    with hoomd.open(name=fname_out, mode="w") as fout:
        fout.append(snap2)

