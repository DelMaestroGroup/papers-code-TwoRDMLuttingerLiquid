import numpy as np
from numpy import pi
from chord import chord_sine
from LL_definitions import gamma_from_K

def one_rdm(xp, x, epsilon, K, L, kF=pi / 2):
    """Evaluate rho_1(xp,x)"""
    delta_x = np.asarray(xp) - np.asarray(x)
    denominator = L * np.sin(pi * delta_x / L)
    numerator = np.sin(kF * delta_x)

    # At delta_x = nL, use l'Hopital's rule for the free finite-size kernel.
    near_removable_zero = np.isclose(denominator, 0.0, atol=1e-13, rtol=0.0)
    winding = np.rint(delta_x / L)
    kernel_limit = ( (kF / pi) * np.cos(kF * winding * L) / np.cos(pi * winding) )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        free_kernel = np.where(near_removable_zero, kernel_limit, numerator / denominator,)
        uv_ratio = (  np.abs(chord_sine(1j * epsilon, 0, L)) / np.abs(chord_sine(xp, x, L, epsilon)) )
        result = free_kernel * uv_ratio ** gamma_from_K(K)

    return result
 