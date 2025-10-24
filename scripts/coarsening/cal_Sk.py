import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
sys.path.append("..")
from scripts.add_line import add_line
from scripts.coarsening.coarse_grain import coarse_grain_density


def orientation_average(S, q_radial, q_module):
    """ Cal orientation-averaged structure factor"""
    Sq = np.zeros_like(q_radial)
    half_dq = (q_radial[1]-q_radial[0]) / 2

    for i, q in enumerate(q_radial):
        mask = np.logical_and(q_module >= q-half_dq, q_module < q+half_dq)
        Sq[i] = np.mean(S[mask])
    return Sq


def get_q1(q_radial, Sq):
    """ Get the first moment of s(q, t). """
    I0 = np.sum(Sq)
    I1 = np.sum(q_radial * Sq)
    q1 = I1 / I0
    return q1


def cal_Sq_t_rho(density_file, dx, dt=None, L=None):
    basename = os.path.basename(density_file)
    if dt is None or L is None:
        s = basename.rstrip(".npz").split("_")
        dt = float(s[8].lstrip("h"))
        L = float(s[0].lstrip("L"))
    folder = f"coarsening/data/sk_dx{dx:g}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    fout = f"{folder}/{basename}"

    if os.path.exists(fout) and (not os.path.exists(density_file)  or os.path.getmtime(fout) > os.path.getmtime(density_file)):
        with np.load(fout) as data:
            t, q_radial, rho_Sqt, rho_var = data["t"], data["q"], data["rho_Sqt"], data["rho_var"]
        print("loading data from", fout)
    else:
        with np.load(density_file, "r") as data:
            t, x, y, rho = data["t"] * dt, data["x"], data["y"], data["density"]
        n = int(L / dx)
        qx = np.fft.fftfreq(n, d=dx/(2 * np.pi))
        qy = np.fft.fftfreq(n, d=dx/(2 * np.pi))
        q_radial = qx[1:qx.size//2-1]
        qx = np.fft.fftshift(qx)
        qy = np.fft.fftshift(qy)
        qx_ij, qy_ij = np.meshgrid(qx, qy)
        q_module = np.sqrt(qx_ij **2 + qy_ij ** 2)
        rho_Sqt = np.zeros((t.size, q_radial.size))
        rho_var = np.zeros(t.size)

        beg = 0
        if os.path.exists(fout):
            with np.load(fout, "r") as data:
                t_pre, q_pre, rho_Sqt_pre, rho_var_pre = data["t"], data["q"], data["rho_Sqt"], data["rho_var"]
                beg = t_pre.size
                rho_Sqt[:beg, :] = rho_Sqt_pre
                rho_var[:beg] = rho_var_pre
                print("loading %d frames from" % beg, fout)

        for j, t_i in enumerate(t[beg:]):
            i = j + beg
            print("processing %d/%d frame, t=%g" % (i, t.size, t_i))
            rho_m = np.mean(rho[i])
            rho_q = np.fft.fft2(rho[i]-rho_m, norm="ortho")
            rho_q = np.fft.fftshift(rho_q)
            S_q = np.abs(rho_q) ** 2
            
            rho_Sqt[i] = orientation_average(S_q, q_radial, q_module)
            rho_var[i] = np.var(rho[i])
        np.savez_compressed(fout, t=t, q=q_radial, rho_Sqt=rho_Sqt, rho_var=rho_var)
        print("updating cache:", fout)
    return t, q_radial, rho_Sqt, rho_var


def update_Sk():
    epyc01 = "/run/user/1000/gvfs/sftp:host=10.10.9.150,user=ps/home/ps/data"
    epyc02 = "/run/user/1000/gvfs/sftp:host=10.10.9.158,user=ps/home/ps/data"
    sli = "/run/user/1000/gvfs/sftp:host=10.10.14.155,port=10066,user=sli/home/sli/yduan"

    folder1 = f"{epyc01}/Fore_aft_QS/Offset_MPI/L240"
    folder2 = f"{epyc02}/Fore_aft_QS/Offset_MPI/L120"
    folder3 = f"{epyc02}/Fore_aft_QS/Offset_MPI/L240"
    folder4 = f"{sli}/Offset_MPI/L120"
    folder5 = f"{sli}/Offset_MPI/L480"

    fgsds = glob.glob(f"{folder1}/*.gsd") + glob.glob(f"{folder2}/*.gsd") + glob.glob(f"{folder3}/*.gsd") + glob.glob(f"{folder4}/*.gsd") + glob.glob(f"{folder5}/*.gsd")
    dx = 0.25
    f_cg = []
    for fgsd in fgsds:
        fout = f"coarsening/data/cg_dx{dx:g}/{os.path.basename(fgsd).replace(".gsd", ".npz")}"
        coarse_grain_density(fgsd, fout, dx=dx)
        f_cg.append(fout)
    
    for f in f_cg:
        cal_Sq_t_rho(f, dx)
    



if __name__ == "__main__":
    update_Sk()
    folder = "coarsening/data/cg_dx0.25"
    
    f_snaps = [f"{folder}/L120_120_Dr2.300_Dt0.000_r80_p20_e3.000_E0.500_h0.025_1000.npz",
               f"{folder}/L120_120_Dr2.300_Dt0.000_r80_p20_e3.000_E0.500_h0.050_1000.npz",
               f"{folder}/L120_120_Dr1.500_Dt0.000_r80_p20_e3.000_E0.500_h0.025_1000.npz",
               f"{folder}/L120_120_Dr1.500_Dt0.000_r80_p20_e3.000_E0.500_h0.050_1000.npz"
               ]
    
    # f_snaps = [f"{folder}/L120_120_Dr2.300_Dt0.000_r40_p10_e3.000_E0.500_h0.050_1000.npz",
    #            f"{folder}/L120_120_Dr2.200_Dt0.000_r40_p10_e3.000_E0.500_h0.050_1000.npz",
    #            f"{folder}/L120_120_Dr2.000_Dt0.000_r40_p8_e3.000_E0.500_h0.100_1000.npz",
    #            f"{folder}/L120_120_Dr2.000_Dt0.000_r40_p6_e3.000_E0.500_h0.100_1000.npz",
    #         #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p8_e3.000_E0.500_h0.100_1000.npz",
    #         #    f"{folder}/L120_120_Dr1.500_Dt0.000_r80_p20_e3.000_E0.500_h0.050_1000.npz"
    #            ]
    # f_snaps = [f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p10_e3.000_E0.500_h0.100_1000.npz",
    #            f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p8_e3.000_E0.500_h0.100_1000.npz",
    #         #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p7_e3.000_E0.500_h0.100_1000.npz",
    #            f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p6_e3.000_E0.500_h0.100_1000.npz",
    #            f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p5_e3.000_E0.500_h0.100_1000.npz"
    #         #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p4_e3.000_E0.500_h0.100_1000.npz"
    #            ]

    # f_snaps = [f"{folder}/L240_240_Dr3.000_Dt0.000_r40_p40_e3.000_E0.500_h0.100_1000.npz",
    #            f"{folder}/L120_120_Dr3.000_Dt0.000_r30_p28.5_e3.000_E0.500_h0.100_1000.npz",
    #            f"{folder}/L480_480_Dr3.000_Dt0.000_r30_p28.5_e3.000_E0.500_h0.100_1000.npz",
    #           ]
    f_snaps = [f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p6_e3.000_E0.500_h0.100_1000.npz",
               f"{folder}/L120_120_Dr3.000_Dt0.000_r30_p4_e3.000_E0.500_h0.100_1000.npz",
               f"{folder}/L120_120_Dr3.000_Dt0.000_r30_p4.5_e3.000_E0.500_h0.100_1000.npz",
               f"{folder}/L120_120_Dr3.000_Dt0.000_r30_p5_e3.000_E0.500_h0.100_1000.npz",
              ]
    f_snaps = [f"{folder}/L240_240_Dr3.000_Dt0.000_r40_p6_e3.000_E0.500_h0.100_1000.npz",
            #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p6_e3.000_E0.500_h0.100_1000.npz",
               f"{folder}/L240_240_Dr3.000_Dt0.000_r40_p40_e3.000_E0.500_h0.100_1000.npz",
            #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p40_e3.000_E0.500_h0.100_1000.npz",
            #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p42_e3.000_E0.500_h0.100_1000.npz",
            #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p44_e3.000_E0.500_h0.050_1000.npz",
            #    f"{folder}/L120_120_Dr3.000_Dt0.000_r40_p45_e3.000_E0.500_h0.050_1000.npz",
                #  f"{folder}/L120_120_Dr2.000_Dt0.000_r40_p8_e3.000_E0.500_h0.100_1000.npz",
                 f"{folder}/L120_120_Dr1.500_Dt0.000_r80_p20_e3.000_E0.500_h0.025_1000.npz"
               ]
    dx = 0.25
    mk = ["o", "s", "p", ">", "*", "<", "+"]
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
    for j, f_snap in enumerate(f_snaps):
        t, q_radial, rho_Sqt, rho_var = cal_Sq_t_rho(f_snap, dx)
        xi = np.zeros(t.size)
        for i in range(t.size):
            xi[i] = 2 * np.pi / get_q1(q_radial, rho_Sqt[i])
        ax.plot(t, xi, mk[j], fillstyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")

    add_line(ax, 0, 0, 1, 1/3)
    add_line(ax, 0.12, 0, 1, 1/3)
    # add_line(ax, 0.1, 0, 1, 1/8)
    add_line(ax, 0, 0, 1, 1/4)
    add_line(ax, 0, 0.05, 1, 1/4)
    # add_line(ax, 0.4, 0, 1, 1/2)
    plt.show()
    plt.close()

