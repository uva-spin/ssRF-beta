"""Complex spin-1 powder susceptibility; Dulya et al., NIM A 398 (1997) 109.

doi:10.1016/S0168-9002(97)00317-3. Normalization: see the physics notes.
"""
from functools import lru_cache
import numpy as np

SIMULATOR = "qmeter-complex-pake-v2"


def complex_branch(x, eps, eta=0.03, phi=0.0, g=0.08):
    """Unit-integral absorption in x; chi = chi' - i chi''.

    Integrate 1/(x_res-x+i*g) over the orientation variable analytically.
    -Im(kernel) integrates to one on the infinite axis, not a finite scan.
    """
    x, phi = np.asarray(x, dtype=float), np.asarray(phi, dtype=float)
    if eps not in (-1, 1) or not np.isfinite([eta, g]).all() or not 0 <= eta <= 1 or g <= 0:
        raise ValueError("Require eps = ±1, 0 <= eta <= 1, and g > 0")
    if not np.isfinite(x).all() or not np.isfinite(phi).all():
        raise ValueError("Frequency and azimuth must be finite")
    y = np.sqrt(3 - eta*np.cos(2*phi))
    z = 1 - eps*x - eta*np.cos(2*phi) + 1j*g
    root = np.sqrt(z)
    integral = np.arctanh(y/root)/root
    if eps == -1:
        integral = -integral.conjugate()
    return integral/(np.pi*y)


@lru_cache(maxsize=16)
def azimuth_quadrature(nphi):
    nodes, weights = np.polynomial.legendre.leggauss(nphi)
    return (nodes + 1)*np.pi/4, weights/2


def powder_complex(x, eps, eta=0.03, g=0.08, nphi=32):
    """Normalized Dulya powder average; Gauss–Legendre azimuth quadrature."""
    if not isinstance(nphi, (int, np.integer)) or nphi < 1:
        raise ValueError("nphi must be a positive integer")
    x = np.asarray(x, dtype=float)
    if eta == 0:
        return complex_branch(x, eps, eta, 0, g)
    nodes, weights = azimuth_quadrature(nphi)
    return (complex_branch(x.reshape(-1, 1), eps, eta, nodes, g) @ weights).reshape(x.shape)


def spin_weights(p):
    """Transition population differences (P+Q)/2 and (P-Q)/2."""
    if not np.isfinite(p) or not -1 <= p <= 1:
        raise ValueError("Polarization must be finite and between -1 and 1")
    q = 3*p*p/(2 + np.sqrt(4 - 3*p*p))
    return (p + q)/2, (p - q)/2


def pake_susceptibility(x, p, eta=0.03, g=0.08, nphi=32):
    """Complex doublet and transitions; integral(-Im(total), dx) = P.

    The circuit's susceptibility_scale_cgs supplies the amplitude coefficient.
    This function never renormalizes a truncated frequency scan.
    """
    wp, wm = spin_weights(p)
    plus = wp*powder_complex(x, 1, eta, g, nphi)
    minus = wm*powder_complex(x, -1, eta, g, nphi)
    return plus + minus, plus, minus
