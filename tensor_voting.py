"""Sec 4.2: tensor encoding of the Gabor responses (Eq 7) and dense tensor voting.

The paper writes 3D tensors (stick / plate / ball). On a 2D image the plate is the 2D stick and the
ball is the 2D ball, so we work with 2x2 tensors:
    curve saliency    = lambda1 - lambda2   (paper: lambda2 - lambda3), tangent = e1
    junction saliency = lambda2             (paper: lambda3)

Voting uses steerable tensor voting (Franken et al. 2006): each stick votes along its osculating
circles with a field that is a sum of complex-harmonic kernels, so dense voting is a handful of FFT
convolutions. The field's angular decay is cos^(2n)(phi) (Franken's steerable approximation of the
c*kappa^2 curvature term of Eq 8) and its radial decay exp(-r^2 / sigma^2) (Eq 8 with s ~ r).
sigma is the only parameter. Our default value is our choice: the paper does not report it.
"""
from math import comb

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from gabor import orientations

SIGMA = 12.0
N_ANGULAR = 8  # cos^16: ~17 deg half-width. Franken's n=2 field is so wide that parallel Gabor bands
               # vote each other into balls (curve fraction on Fig 3a lines: 19% at n=2, 93% at n=8).


def encode(R):
    """Eq 7: T = sum_i R_i t_i t_i^T with t_i the unit tangent of filter i. Returns (Txx, Txy, Tyy)."""
    th = orientations(len(R))[:, None, None]
    c, s = np.cos(th), np.sin(th)
    return (R * c * c).sum(0), (R * c * s).sum(0), (R * s * s).sum(0)


def decompose(Txx, Txy, Tyy):
    """2x2 tensor -> (stick saliency l1-l2, ball saliency l2, tangent angle of e1 in [0, pi))."""
    half_tr = (Txx + Tyy) / 2
    w = np.hypot((Txx - Tyy) / 2, Txy)
    l1, l2 = half_tr + w, np.maximum(half_tr - w, 0)
    theta = np.mod(0.5 * np.arctan2(2 * Txy, Txx - Tyy), np.pi)
    theta[theta >= np.pi] = 0  # fmod rounding can return pi itself
    return l1 - l2, l2, theta


def _kernels(sigma, n):
    r = int(np.ceil(3 * sigma / np.sqrt(2)))  # exp(-r^2/sigma^2) < 1e-4 beyond
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    rad2 = x * x + y * y
    decay = np.exp(-rad2 / sigma**2)
    psi = np.arctan2(y, x)

    def harmonic(k):
        h = decay * np.exp(1j * k * psi)
        if k:
            h[r, r] = 0  # direction undefined at the voter itself
        return h

    return harmonic


def vote(stick, ball, theta, sigma=SIGMA, n=N_ANGULAR):
    """Dense voting. Inputs per pixel: stick saliency, ball saliency, stick tangent angle.

    A stick at y with tangent beta and saliency mu votes at x = y + r e^{i psi}:
        magnitude mu * exp(-r^2/sigma^2) * cos^{2n}(psi - beta), orientation 2 psi - beta.
    Expanding cos^{2n} into harmonics separates voter and receiver terms:
        U(x)   = sum_j c_j/2 * [mu e^{-2i(j+1)beta}] * [D e^{i(2j+4)psi}]   (deviatoric part)
        tr(x)  = sum_j c_j   * [mu e^{-2ij beta}]   * [D e^{i 2j psi}]       (trace)
    with c_j = C(2n, n+j) / 4^n. A ball of saliency b equals sticks of saliency 2b averaged over
    all beta, which leaves only the j = -1 term of U and the j = 0 term of tr.
    Returns accumulated (Txx, Txy, Tyy).
    """
    harmonic = _kernels(sigma, n)
    U = np.zeros(stick.shape, np.complex128)
    tr = np.zeros(stick.shape, np.float64)
    e2 = np.exp(-2j * theta)
    for j in range(-n, n + 1):
        c = comb(2 * n, n + j) / 4**n
        a = stick * e2 ** (j + 1)
        t = stick * e2**j
        if j == -1:
            a = a + 2 * ball
        if j == 0:
            t = t + 2 * ball
        U += c / 2 * fftconvolve(a, harmonic(2 * j + 4), mode="same")
        tr += c * fftconvolve(t, harmonic(2 * j), mode="same").real
    return tr / 2 + U.real, U.imag, tr / 2 - U.real


def curve_nms(sal, theta):
    """Keep pixels whose saliency is a local max across the curve (along the normal of theta)."""
    h, w = sal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = -np.sin(theta), np.cos(theta)
    a = map_coordinates(sal, [yy + ny, xx + nx], order=1, mode="nearest")
    b = map_coordinates(sal, [yy - ny, xx - nx], order=1, mode="nearest")
    # Floor at 1e-3 of max only removes FFT round-off ridges in empty areas (e.g. Fig 3a background).
    return (sal >= a) & (sal >= b) & (sal > 1e-3 * sal.max())


def infer(R, sigma=SIGMA, n=N_ANGULAR):
    """Gabor responses -> tensor voting -> classification (Sec 4.2). No thresholds.

    Returns dict:
        curve_sal, junction_sal, theta  dense maps after voting
        is_curve     (l1 - l2) > l2  (else junction)
        curve_points is_curve and a saliency ridge across the curve (Fig 3d)
        curve_map    curve_sal on curve_points, normalised to [0, 1]; 0 elsewhere
    """
    stick, ball, theta0 = decompose(*encode(R))
    # Only the stick part votes. The ball part of a Gabor tensor is isotropic texture response; letting
    # it vote swamps every road with junction votes (curve on only 5% of Fig 3b). With stick votes,
    # lambda2 after voting measures disagreement between incoming orientations, i.e. junction-ness.
    s, b, theta = decompose(*vote(stick, np.zeros_like(ball), theta0, sigma, n))
    is_curve = s > b
    pts = is_curve & curve_nms(s, theta)
    cmap = np.where(pts, s, 0) / max(s[pts].max() if pts.any() else 1, 1e-12)
    return dict(curve_sal=s, junction_sal=b, theta=theta, is_curve=is_curve,
                curve_points=pts, curve_map=cmap.astype(np.float32))
