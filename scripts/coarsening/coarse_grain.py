import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from gsd import hoomd

sys.path.append("..")
from scripts.read_snap import get_frames, get_nframe_L


def coarse_grain_density_one_frame(snap: hoomd.Frame,
                                   dx: float,
                                   Lx: float,
                                   Ly: float):
    ncols = int(Lx / dx)
    nrows = int(Ly / dx)
    bin_area = dx ** 2

    par_num = np.zeros((nrows, ncols), int)
    pos = snap.particles.position
    x = pos[:,0] + Lx / 2
    y = pos[:,1] + Ly / 2

    col = (x/dx).astype(int)
    row = (y/dx).astype(int)
    col[col < 0] += ncols
    col[col >= ncols] -= ncols
    row[row < 0] += nrows
    row[row >= nrows] -= nrows
    
    for j in range(x.size):
        my_row = row[j]
        my_col = col[j]
        par_num[my_row, my_col] += 1
    return  par_num / bin_area


def coarse_grain_density(fin: str, fout: str, dx: float):
    if os.path.exists(fout):
        with np.load(fout, "r") as data:
            t0 = data["t"]
            x0 = data["x"]
            y0 = data["y"]
            density_fields0 = data["density"]
            existed_frames = t0.size
    else:
        existed_frames = 0
    
    n_frames, Lx, Ly = get_nframe_L(fin)
    if existed_frames < n_frames:
        ncols = int(Lx / dx)
        nrows = int(Ly / dx)
        x = np.arange(ncols) * dx + dx / 2
        y = np.arange(nrows) * dx + dx / 2
        t = np.zeros(n_frames, int)
        density_fields = np.zeros((n_frames, nrows, ncols), np.single)
        if existed_frames > 0:
            t[:existed_frames] = t0
            density_fields[:existed_frames] = density_fields0
        frames = get_frames(fin, beg_frame=existed_frames)
        for i, frame in enumerate(frames):
            i_frame = i + existed_frames
            t[i_frame] = frame.configuration.step
            density_fields[i_frame] = coarse_grain_density_one_frame(frame, dx, Lx, Ly)

        np.savez_compressed(fout, t=t, x=x, y=y, density=density_fields)




if __name__ == "__main__":
    epyc01 = "/run/user/1000/gvfs/sftp:host=10.10.9.150,user=ps/home/ps/data"
    epyc02 = "/run/user/1000/gvfs/sftp:host=10.10.9.158,user=ps/home/ps/data"
    folder = f"{epyc01}/Fore_aft_QS/Offset_MPI/L120"

    fins = [f"{folder}/L120_120_Dr1.500_Dt0.000_r80_p20_e3.000_E0.500_h0.025_1000.gsd",
            f"{folder}/L120_120_Dr1.500_Dt0.000_r80_p20_e3.000_E0.500_h0.050_1000.gsd"
            ]

    dx = 0.25
    for fin in fins:
        fout = f"coarsening/data/cg_dx{dx:g}/{os.path.basename(fin).replace(".gsd", ".npz")}"

        coarse_grain_density(fin, fout, dx=dx)