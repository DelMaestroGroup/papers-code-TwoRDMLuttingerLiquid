from pathlib import Path
import numpy as np

folder = Path("../data/dmrg/")
def path_obdm(L,V,folder=folder):
    return folder.joinpath(f"obdm_M{L:d}_N{L//2:d}_t+1.000_Vp+0.00000_Vsta{V:+3.5f}_Vend{V:+3.5f}_Vnum0001_V{V:4.4f}.npy") 
 
def load_obdm(L,V,folder=folder):
    return np.load(path_obdm(L,V,folder=folder)) 

def path_tbdm(L,V):
    return folder.joinpath(f"tbdm_M{L:d}_N{L//2:d}_t+1.000_Vp+0.000_Vsta{V:+3.3f}_Vend{V:+3.3f}_Vnum0001_V{V:4.4f}.npy")

def load_tbdm(L,V):
    return np.load(path_tbdm(L,V))

def path_g2(L, N, V):
    """Return the expected DMRG density-correlation filename."""
    filename = f"g2_L{L}_N{N}_J+1.000_V{V:+3.3f}_Vp+0.000_pbc.dat" 
    return folder.joinpath(filename) 

def load_g2(L, N, V):
    """Load and validate the first column of a length-L DMRG g2 file."""
    path = path_g2(L, N, V)  
    raw = np.loadtxt(path)
    values = raw if raw.ndim == 1 else raw[:, 0] 
    return values

def load_energies(directory:Path):
    filenames = [p for p in directory.iterdir() if p.is_file() and f"groundstate_energy" in p.stem]
    Vs, Es = [], []
    for fn in filenames:
        data = np.loadtxt(fn)
        if len(data) == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1,-1)
        Vs.append(data[:,0])
        Es.append(data[:,1])
    return np.array(Vs).reshape(-1), np.array(Es).reshape(-1)