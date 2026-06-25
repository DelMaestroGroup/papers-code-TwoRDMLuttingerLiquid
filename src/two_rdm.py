import numpy as np
from numpy import pi
from LL_definitions import lambda_from_K, gamma_from_K
from one_rdm import one_rdm
from chord import chord_sine, abs_chord_sine, periodic_coincidence


def fixed_delta_R_coordinates(r, rp, delta_R):
    """Map (r,r',Delta R) to coordinates with fixed pair-center separation."""
    x1 = 0.0
    x2 = r
    x1p = delta_R + (r - rp) / 2
    x2p = delta_R + (r + rp) / 2
    return x2p, x1p, x2, x1

def direct_coordinates(r, rp, d):
    """Map (r,r',d=x1'-x1) to (x2',x1',x2,x1), setting x1=0."""
    x1 = 0.0
    x2 = r
    x1p = d
    x2p = d + rp
    return x2p, x1p, x2, x1


def density_density_limit_scalar(delta_x, L, epsilon, K, rho0=0.5):
    """Density-density branch needed when x2'=x2 in the diagonal limit. 
    """
    if periodic_coincidence(delta_x, L):
        return 0.0

    gamma = gamma_from_K(K)
    lam = lambda_from_K(K)
    delta = gamma - lam

    particle_number = int(np.round(rho0 * L))
    kF = pi * rho0 if particle_number % 2 == 0 else pi * (particle_number - 1) / L

    a_epsilon = abs_chord_sine(delta_x, L, epsilon)
    a_zero = abs_chord_sine(delta_x, L, 0.0)
    a_origin = abs_chord_sine(0.0, L, epsilon)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        smooth_part = a_epsilon ** (-2 * delta) / (2 * L**2) * (
            -( a_epsilon ** (2 * delta) - a_origin ** (2 * delta) ) / a_zero**2
            + delta / 2 * a_epsilon ** (2 * delta - 4)
            * ( 2 * a_epsilon**2 * np.cos(2 * pi * delta_x / L) - abs_chord_sine(2 * delta_x, L, 0.0)  ** 2
            )
        )
        oscillatory_part = -(  a_origin ** (2 * delta) / a_epsilon ** (2 * delta) * np.sin(kF * delta_x) ** 2 / (L**2 * a_zero**2) )

    return smooth_part + rho0**2 + oscillatory_part

def two_rdm_fixed_delta_R(r, rp, delta_R, epsilon, K, L, eta=0.0):
    """Full 2RDM at fixed pair-center separation Delta R."""
    return two_rdm(
        *fixed_delta_R_coordinates(r, rp, delta_R), epsilon, K, L, eta
    )
 
def two_rdm(x2p, x1p, x2, x1, epsilon, K, L, eta=0.0):
    """Evaluate rho_2(x2',x1';x2,x1) by summing the three channels."""
    t1, t2, t3 = two_rdm_terms(x2p, x1p, x2, x1, epsilon, K, L, eta)
    return t1 + t2 + t3

def two_rdm_terms(x2p, x1p, x2, x1, epsilon, K, L, eta=0.0):
    """Return the three contributions (t1,t2,t3) to rho_2."""
    phases, envelopes = two_rdm_channel_envelopes(
        x2p, x1p, x2, x1, epsilon, K, L, eta
    )
    theta_1, theta_2, theta_3 = phases
    envelope_1, envelope_2, envelope_3 = envelopes

    t1 = -np.cos(theta_1) * envelope_1 / (2 * L**2)
    t2 = -np.cos(theta_2) * envelope_2 / (2 * L**2)
    t3 = +np.cos(theta_3) * envelope_3 / (2 * L**2)
    return t1, t2, t3

def two_rdm_channel_envelopes(x2p, x1p, x2, x1, epsilon, K, L, eta=0.0):
    """Return the three Fermi phases and three bosonization envelopes."""
    gamma = gamma_from_K(K)
    lam = lambda_from_K(K)
    kF = pi / 2
    s_epsilon = chord_sine(1j * epsilon, 0, L)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"): 
        envelope_1 = (
            chord_sine(x2p, x1p, L) * chord_sine(x2, x1, L)  / (
                chord_sine(x1p, x2, L, eta) * chord_sine(x1p, x1, L, eta) * chord_sine(x2p, x2, L, eta) * chord_sine(x2p, x1, L, eta)
            )
        )
        envelope_1 *= np.abs(
            s_epsilon**2 * chord_sine(x2p, x1p, L, epsilon)  * chord_sine(x2, x1, L, epsilon)
            / ( chord_sine(x2p, x2, L, epsilon) * chord_sine(x1p, x1, L, epsilon) * chord_sine(x2p, x1, L, epsilon) * chord_sine(x1p, x2, L, epsilon)
            )
        ) ** gamma

        # Channel 2 pairs x2' with x2 and x1' with x1 in its bare poles.
        envelope_2 = 1 / ( chord_sine(x2p, x2, L, eta) * chord_sine(x1p, x1, L, eta) )
        envelope_2 *= np.abs(
            s_epsilon**2 / ( chord_sine(x2p, x2, L, epsilon) * chord_sine(x1p, x1, L, epsilon)  )
        ) ** gamma
        envelope_2 *= np.abs(
              chord_sine(x2p, x1p, L, epsilon) * chord_sine(x2, x1, L, epsilon) / (  chord_sine(x2p, x1, L, epsilon) * chord_sine(x1p, x2, L, epsilon) )
        ) ** lam

        # Channel 3 pairs x2' with x1 and x1' with x2 in its bare poles.
        envelope_3 = 1 / (  chord_sine(x2p, x1, L, eta) * chord_sine(x1p, x2, L, eta) )
        envelope_3 *= np.abs( s_epsilon**2 / ( chord_sine(x2p, x1, L, epsilon) * chord_sine(x1p, x2, L, epsilon) ) ) ** gamma
        envelope_3 *= np.abs( chord_sine(x2p, x1p, L, epsilon) * chord_sine(x2, x1, L, epsilon) / (  chord_sine(x2p, x2, L, epsilon) * chord_sine(x1p, x1, L, epsilon) )
        ) ** lam

    theta_1 = kF * (x1 + x2 - x1p - x2p)
    theta_2 = kF * (x1p + x2 - x1 - x2p)
    theta_3 = kF * (x1p + x1 - x2p - x2)

    return (theta_1, theta_2, theta_3), (envelope_1, envelope_2, envelope_3)


def wick_residual_direct(r, rp, d, epsilon, K, L, eta=0.0):
    """rho_2-A[rho_1 rho_1] on the direct (r,r',d) slice."""
    return wick_residual(*direct_coordinates(r, rp, d), epsilon, K, L, eta)
 
def wick_two_rdm_from_one_rdm(x2p, x1p, x2, x1, epsilon, K, L):
    """Antisymmetrized product of two 1RDMs at the same value of K."""
    direct = one_rdm(x2p, x1, epsilon, K, L) * one_rdm(x1p, x2, epsilon, K, L)
    exchange = one_rdm(x2p, x2, epsilon, K, L) * one_rdm(x1p, x1, epsilon, K, L)
    return direct - exchange

def wick_residual(x2p, x1p, x2, x1, epsilon, K, L, eta=0.0):
    """Return rho_2 - A[rho_1 rho_1] at fixed interacting K."""
    return two_rdm(x2p, x1p, x2, x1, epsilon, K, L, eta) - wick_two_rdm_from_one_rdm(x2p, x1p, x2, x1, epsilon, K, L)

def interaction_correction_direct(r, rp, d, epsilon, K, L, eta=0.0):
    """rho_2(K)-rho_2(1) on the direct (r,r',d) slice."""
    return interaction_correction(*direct_coordinates(r, rp, d), epsilon, K, L, eta)

def interaction_correction(x2p, x1p, x2, x1, epsilon, K, L, eta=0.0):
    """Return rho_2(K) - rho_2(K=1)."""
    return two_rdm(x2p, x1p, x2, x1, epsilon, K, L, eta) - free_two_rdm(x2p, x1p, x2, x1, epsilon, L, eta)

def free_two_rdm(x2p, x1p, x2, x1, epsilon, L, eta=0.0):
    """Exact free-fermion reference obtained from the full formula at K=1."""
    return two_rdm(x2p, x1p, x2, x1, epsilon, 1.0, L, eta)


def analytic_g2(r, epsilon, K, L, rho0, kF):
    """Evaluate the normalized diagonal-2RDM pair distribution g2(r)."""  

    delta = K - 1.0
    a_eps = abs_chord_sine(r, L, epsilon)
    a_zero = abs_chord_sine(r, L, 0.0)
    a_eps_origin = abs_chord_sine(0.0, L, epsilon)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        smooth = a_eps ** (-2.0 * delta) / (2.0 * L**2) * (
            -(a_eps ** (2.0 * delta) - a_eps_origin ** (2.0 * delta)) / a_zero**2
            + 0.5 * delta * a_eps ** (2.0 * delta - 4.0)
            * (
                2.0 * a_eps**2 * np.cos(2.0 * np.pi * r / L)
                - abs_chord_sine(2.0 * r, L, 0.0) ** 2
            )
        )
        oscillatory = -(
            a_eps_origin ** (2.0 * delta)
            / a_eps ** (2.0 * delta)
            * np.sin(kF * r) ** 2
            / (L**2 * a_zero**2)
        )
        result = (rho0**2 + smooth + oscillatory) / rho0**2

    # Normal ordering and Pauli exclusion require a zero at r = 0 mod L.
    coincidence = np.isclose(
        np.sin(np.pi * r / L), 0.0, atol=1.0e-13, rtol=0.0
    )
    result = np.where(coincidence, 0.0, np.real(result))
    return result.item() if result.ndim == 0 else result
