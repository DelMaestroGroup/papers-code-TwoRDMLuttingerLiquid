import numpy as np
from one_rdm import one_rdm
from two_rdm import density_density_limit_scalar
from chord import abs_chord_sine, dlog_regularized_chord_abs
from LL_definitions import gamma_from_K, lambda_from_K

def delta_rho_cdw(xi, epsilon, K, L=40.0):
    r"""
    Evaluate the background-subtracted CDW coherence

        delta rho(xi) = rho(xi) + rho0 rho1(xi). 
    """
    rho0 =  np.real( one_rdm(0.0, 0.0, epsilon, K, L)  ) 
    rho1_xi = np.real( one_rdm(xi, 0.0, epsilon, K, L) )

    return rho_cdw_raw( xi, epsilon, K, L=L, rho0=rho0, ) + rho0 * rho1_xi
    

def rho_cdw_raw(xi, epsilon, K, L=40.0, rho0=0.5):
    r"""
    Evaluate the raw CDW off-diagonal coherence

        rho(xi) = <Psi^dagger(0) Psi^dagger(R+xi) Psi(0) Psi(R)>,

    where R=L/2. 
    """
    if not np.isclose(rho0, 0.5):
        raise ValueError( "The notebook's 2RDM expression is specialized to half filling."  )

    xi_array = np.asarray(xi, dtype=float)
    scalar_input = xi_array.ndim == 0
    x = np.atleast_1d(xi_array)

    R = L / 2.0
    p = np.pi / L
    kF = np.pi / 2.0

    gamma = gamma_from_K(K)
    lam = lambda_from_K(K)
    delta = gamma - lam       # exactly K-1

    result = np.empty_like(x, dtype=float)

    # xi=0 modulo L: 
    # <Psi†(0)Psi†(R)Psi(0)Psi(R)> = -<n(0)n(R)>.
    diagonal = np.isclose( np.sin(p * x), 0.0, atol=1.0e-13, rtol=0.0, )

    if np.any(diagonal):
        result[diagonal] = -density_density_limit_scalar( R, L, epsilon, K, rho0=rho0,)

    # R+xi=0 modulo L makes the two creation coordinates coincide.
    # The corresponding 2RDM element vanishes by Pauli exclusion.
    pauli = (~diagonal) & np.isclose( np.sin(p * (R + x)), 0.0, atol=1.0e-13, rtol=0.0, )

    result[pauli] = 0.0

    regular = ~(diagonal | pauli)

    if np.any(regular):
        xx = x[regular]

        # After simultaneously exchanging the primed and unprimed particle
        # labels, the coincidence limit can be taken as x1p -> x1=0: 
        # rho(0,R+xi;0,R) = rho(R+xi,0;R,0).
        A = R + xx
        B = R

        h0 = abs_chord_sine(0.0, L=L, epsilon=epsilon)
        hx = abs_chord_sine(xx, L=L, epsilon=epsilon)
        hA = abs_chord_sine(A, L=L, epsilon=epsilon)
        hB = abs_chord_sine(B, L=L, epsilon=epsilon)

        # Finite limit of channels 1 and 2. Individually, these channels
        # contain poles proportional to 1/sin(pi q/L), with q=x1p-x1.
        # Their residues cancel, leaving the derivative below.
        amplitude =  (h0 / hx)**gamma  / np.sin(p * xx) 

        logarithmic_derivative =  delta * dlog_regularized_chord_abs(A, L=L, epsilon=epsilon) - dlog_regularized_chord_abs(B, L=L, epsilon=epsilon) + p * ( np.cos(p * A) / np.sin(p * A) - np.cos(p * B) / np.sin(p * B) )
        

        term_12 = ( -amplitude / (2.0 * L**2 * p)  * (
                2.0 * kF * np.sin(kF * xx) + np.cos(kF * xx) * logarithmic_derivative
            )
        )

        # Channel 3 remains finite without differentiation.
        envelope_3 =  -1.0 / ( np.sin(p * A) * np.sin(p * B) ) * ( h0**2 / (hA * hB) )**gamma * ( hA * hB / (hx * h0) )**lam 

        term_3 =   np.cos(kF * (A + B)) * envelope_3  / (2.0 * L**2)   

        result[regular] = term_12 + term_3

    return result[0] if scalar_input else result