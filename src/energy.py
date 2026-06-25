import numpy as np
from LL_definitions import gamma_from_V, K_from_V
from two_rdm import analytic_g2
from one_rdm import one_rdm

def Ekin_sum(ϵ, V, L): 
    kF= np.pi/2 
    K = K_from_V(V)
    rho1 = one_rdm(1, 0, epsilon=ϵ, K=K, L=L, kF=kF)
    rhom1 = one_rdm(-1, 0, epsilon=ϵ, K=K, L=L, kF=kF)
    return  -L * (rho1 + rhom1)

def Eint_sum(ϵ, V, L):  
    N = L/2
    ρ0 = N/L
    kF = np.pi * ρ0
    K = K_from_V(V)
    g2 = analytic_g2(np.array(range(L)), epsilon=ϵ, K=K, L=L, rho0=ρ0, kF=kF) 
    Δg = g2.sum() - (L-1/ρ0)  
 
    return V * L * (g2[1] - Δg/2) * ρ0**2

def E_sum(ϵ, V, L):
    return Ekin_sum(ϵ, V, L) + Eint_sum(ϵ, V, L) 