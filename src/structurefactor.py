import numpy as np

def structure_factor(g2_values, rho0):
    """Compute S(q) on q=2*pi*m/L using NumPy's unnormalized rFFT."""  

    q = 2.0 * np.pi * np.fft.rfftfreq(g2_values.size, d=1.0)
    S = 1.0 + rho0 * np.fft.rfft(g2_values - 1.0)
    return q, S

def enforce_pair_sum_rule(g2_values, rho0):
    """Enforce S(0)=0 by a symmetric shift at r=+/-1.  """
    corrected = np.asarray(g2_values, dtype=float).copy() 

    target = corrected.size - 1.0 / rho0
    correction = target - np.sum(corrected)
    corrected[1] += 0.5 * correction
    corrected[-1] += 0.5 * correction
    return corrected, correction, target