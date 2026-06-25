import numpy as np

def K_from_V(V):
    """Return the Luttinger parameter K for -2 < V/J <= 2.""" 
    return np.pi / (2 * np.arccos(-V / 2))

def lambda_from_K(K):
    """Return lambda = (1/K - K)/2.""" 
    return 0.5 * (1 / K - K)

def lambda_from_V(V):
    """Return lambda = (1/K - K)/2.""" 
    K = K_from_V(V)
    return lambda_from_K(K)

def gamma_from_K(K):
    """Return gamma = (K + 1/K - 2)/2.""" 
    return 0.5 * (K + 1 / K - 2)

def gamma_from_V(V):
    """Return gamma = (K + 1/K - 2)/2.""" 
    K = K_from_V(V)
    return gamma_from_K(K)
