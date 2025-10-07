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


def get_frame(fname_in, i_frame=-1, flag_show=False):
    i_frame = -1
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


if __name__ == "__main__":
    pass