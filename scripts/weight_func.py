import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import factorial


def offset_kernel(psi, r, eps, l, m):
    R = np.sqrt(r**2 - 2 * eps * r * np.cos(psi) + eps**2)
    if R < 1:
        prefactor  = 1 / (2 ** l * factorial(m, exact=True) * factorial(l-m, exact=True))
        res = prefactor * 3 / np.pi * (1 - R) * r**(l+1) * np.cos((l-2*m) * psi)
    else:
        res = 0
    return res


def VC_kernel(psi, r, half_angle, l, m):
    if r < 1:
        prefactor  = 1 / (2 ** l * factorial(m, exact=True) * factorial(l-m, exact=True))
        res = prefactor * 3 / half_angle * (1 - r) * r**(l+1) * np.cos((l-2*m) * psi)
        # res = prefactor * 1 / half_angle  * r**(l+1) * np.cos((l-2*m) * psi)
    else:
        res = 0
    return res


if __name__ == "__main__":
    l = 1
    m = 0
    # eps_arr = np.linspace(0, 0.5, 50)
    # w10_arr = np.zeros_like(eps_arr)

    # for i, eps in enumerate(eps_arr):
    #     I = integrate.dblquad(offset_kernel, 0, 1+eps, 0, 2 * np.pi, args=(eps, l, m))
    #     w10_arr[i] = I[0]
    
    # plt.plot(eps_arr, w10_arr)
    # plt.plot(eps_arr, eps_arr * 0.5)
    # plt.show()
    # plt.close()


    alpha_arr = np.pi - np.linspace(0, np.pi, 60, endpoint=False)
    w10_arr = np.zeros_like(alpha_arr)


    for i, alpha in enumerate(alpha_arr):
        I = integrate.dblquad(VC_kernel, 0, 1, -alpha, alpha, args=(alpha, l, m))
        w10_arr[i] = I[0]
    
    plt.plot(alpha_arr, w10_arr)
    plt.axhline(1/4, linestyle="dashed")
    plt.axhline(1/8, linestyle="dashed")
    plt.axvline(np.pi/2)
    plt.axvline(1.275535, linestyle="dotted")
    plt.show()
    plt.close()