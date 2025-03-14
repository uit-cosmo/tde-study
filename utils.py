import numpy as np


def taumax(dx, dy, lpara, lperp, v, w, a):
    d1 = (dx * lperp**2 * v + dy * lpara**2 * w) * np.cos(a) ** 2
    d2 = (dx * lpara**2 * v + dy * lperp**2 * w) * np.sin(a) ** 2
    d3 = -(lpara**2 - lperp**2) * (dy * v + dx * w) * np.cos(a) * np.sin(a)
    n1 = (lperp**2 * v**2 + lpara**2 * w**2) * np.cos(a) ** 2
    n2 = -2 * (lpara**2 - lperp**2) * v * w * np.sin(a) * np.cos(a)
    n3 = (lpara**2 * v**2 + lperp**2 * w**2) * np.sin(a) ** 2
    return (d1 + d2 + d3) / (n1 + n2 + n3)


def v3(lpara, lperp, v, w, a):
    tx = taumax(1, 0, lpara, lperp, v, w, a)
    ty = taumax(0, 1, lpara, lperp, v, w, a)
    return tx / (tx**2 + ty**2)


def w3(lpara, lperp, v, w, a):
    tx = taumax(1, 0, lpara, lperp, v, w, a)
    ty = taumax(0, 1, lpara, lperp, v, w, a)
    return ty / (tx**2 + ty**2)
