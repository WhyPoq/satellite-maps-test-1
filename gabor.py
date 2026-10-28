"""Sec 4.1: Gabor filter bank (8 orientations x 5 frequencies, Manjunath & Ma 1996 parameters)."""
import cv2
import numpy as np
from skimage.filters import gabor_kernel

N_ORIENT = 8
N_SCALE = 5
UL, UH = 0.05, 0.4  # lower / upper centre frequency (cycles per pixel)


def manjunath_ma_params(ul=UL, uh=UH, n_scale=N_SCALE, n_orient=N_ORIENT):
    """Return [(frequency, sigma_x, sigma_y)] per scale, finest scale first.

    sigma_x is along the carrier (across the feature), sigma_y along the feature.
    """
    a = (uh / ul) ** (1.0 / (n_scale - 1))
    ln2 = np.log(2.0)
    su = (a - 1) * uh / ((a + 1) * np.sqrt(2 * ln2))
    sv = np.tan(np.pi / (2 * n_orient)) * (uh - 2 * ln2 * su**2 / uh) / np.sqrt(
        2 * ln2 - (2 * ln2) ** 2 * su**2 / uh**2)
    sx, sy = 1 / (2 * np.pi * su), 1 / (2 * np.pi * sv)
    # Dilating the mother wavelet by a^m divides the frequency and multiplies the spatial sigmas.
    return [(uh / a**m, sx * a**m, sy * a**m) for m in range(n_scale)]


def orientations(n=N_ORIENT):
    """Feature (tangent) orientations theta_i = i*pi/n in [0, pi)."""
    return np.arange(n) * np.pi / n


def gabor_bank(gray, n_orient=N_ORIENT, n_scale=N_SCALE, ul=UL, uh=UH):
    """Gabor response magnitudes summed over frequency: R[i] for tangent orientation theta_i.

    A line or edge with tangent theta varies across theta + pi/2, so the carrier of filter i points
    along theta_i + pi/2. Output is normalised to max 1 (Table 1, last column).
    """
    gray = gray.astype(np.float32)
    R = np.zeros((n_orient,) + gray.shape, np.float32)
    for i, theta in enumerate(orientations(n_orient)):
        for f, sx, sy in manjunath_ma_params(ul=ul, uh=uh, n_scale=n_scale, n_orient=n_orient):
            k = gabor_kernel(f, theta=theta + np.pi / 2, sigma_x=sx, sigma_y=sy, n_stds=3)
            re = (k.real - k.real.mean()).astype(np.float32)  # zero DC: no response to flat brightness
            im = k.imag.astype(np.float32)
            r = cv2.filter2D(gray, -1, re, borderType=cv2.BORDER_REFLECT)
            q = cv2.filter2D(gray, -1, im, borderType=cv2.BORDER_REFLECT)
            R[i] += np.sqrt(r * r + q * q)
    return R / max(R.max(), 1e-12)
