import matplotlib.pyplot as plt
from gsd import hoomd, fl
import sys


def get_one_snap(f, i_frame):
    s = hoomd.Frame()
    s.configuration.box = f.read_chunk(frame=0, name="configuration/box")
    # s.particles.types = f.read_chunk(frame=0, name="particles/types")
    # try:
    #     if not isinstance(s.particles.types[0], str):
    #         s.particles.types = [chr(i) for i in s.particles.types]
    # except TypeError:
    #     s.particles.types = ['A', 'B']

    if f.chunk_exists(frame=i_frame, name="configuration/step"):
        s.configuration.step = f.read_chunk(frame=i_frame,
                                            name="configuration/step")[0]
        print(s.configuration.step)
    else:
        if i_frame == 0:
            s.configuration.step = 0
        else:
            print("Error, cannot find step for frame =", i_frame)
            sys.exit()
    s.particles.N = f.read_chunk(frame=i_frame, name="particles/N")[0]
    # if f.chunk_exists(frame=i_frame, name="particles/typeid"):
    #     s.particles.typeid = f.read_chunk(frame=i_frame, name="particles/typeid")
    # else:
    #     s.particles.typeid = np.zeros(s.particles.N, np.int32)
    """"
    position = [x, y, theta]
    x in [-Lx/2, Lx/2)
    y in [-Ly/2, Ly/2]
    theta in [-PI, PI]
    """
    s.particles.position = f.read_chunk(frame=i_frame,
                                        name="particles/position")
    s.particles.charge = f.read_chunk(frame=i_frame, name="particles/charge")
    return s


def read_one_frame(fname, i_frame):
    with fl.open(name=fname, mode="r") as f:
        if f.nframes == 0:
            print("Warning, zero frame found in", fname)
            return None
        else:
            if i_frame < 0:
                i = i_frame + f.nframes
            else:
                i = i_frame
            print("read frame %d in total %d frames" % (i, f.nframes))
            return get_one_snap(f, i)


def get_nframe(fname):
    with fl.open(name=fname, mode="r") as f:
        return f.nframes


def get_nframe_L(fname):
    with fl.open(name=fname, mode="r") as f:
        Lx, Ly = f.read_chunk(frame=0, name="configuration/box")[:2]
        return f.nframes, Lx, Ly 


def get_frame(fname_in, i_frame=-1, flag_show=False):
    try:
        s = hoomd.Frame()
        with hoomd.open(name=fname_in, mode='r') as fin:
            nframes = len(fin)
            print(nframes)
            if i_frame < 0:
                i_frame += nframes
            s = fin[i_frame]
    except IndexError:
        s = read_one_frame(fname_in, i_frame)
    if flag_show:
        x = s.particles.position[:, 0]
        y = s.particles.position[:, 1]
        # theta = s.particles.charge
        # plt.imshow(theta.reshape(512, 512), origin="lower")
        plt.plot(x, y, ".", ms=1)
        plt.show()
        plt.close()
    return s


def get_frames(fname_in, beg_frame=0, end_frame=None, flag_show=False):
    try:
        s = hoomd.Frame()
        with hoomd.open(name=fname_in, mode='r') as fin:
            nframes = len(fin)
            if end_frame is None:
                end_frame = nframes
            elif end_frame < 0:
                end_frame += nframes
            for i in range(beg_frame, end_frame):
                yield fin[i]
    except IndexError:
        with fl.open(name=fname_in, mode="r") as f:
            if end_frame is None:
                end_frame = f.nframes
            elif end_frame < 0:
                end_frame += nframes
            for i in range(beg_frame, end_frame):
                yield get_one_snap(f, i)


if __name__ == "__main__":
    epyc01 = "/run/user/1000/gvfs/sftp:host=10.10.9.150,user=ps/home/ps/data"
    folder = f"{epyc01}//Offset_negative/L21_r160_e-0.75"
    fname = f"{folder}/L21_21_Dr0.010_Dt0.000_r160_p160_e-0.500_E-0.750_h0.050_1000.gsd"

    frames = get_frames(fname)
    pass
