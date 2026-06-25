import numpy as np
from numpy import pi

def chord_sine(x, y, L, regulator=0.0):
    """Dimensionless periodic chord sine s(x,y;regulator). """
    return np.sin(pi / L * (np.asarray(x) - np.asarray(y) + 1j * regulator))

def abs_chord_sine(delta_x, L, epsilon=0.0):
    """Absolute value of sin[pi(delta_x+i epsilon)/L]."""
    return np.abs(np.sin(pi / L * (delta_x + 1j * epsilon)))

def dlog_regularized_chord_abs(x, L, epsilon):
    """Return d/dx log|sin[pi (x+i epsilon)/L]|."""
    p = np.pi / L 

    denominator = np.sin(p * x)**2 + np.sinh(p * epsilon)**2 

    return  p * np.sin(p * x) * np.cos(p * x)  / denominator 

def periodic_coincidence(delta_x, L, tolerance=1e-11):
    """Test whether a scalar separation is zero modulo the ring length."""
    winding = round(abs(delta_x) / L)
    return abs(abs(delta_x) - winding * L) < tolerance

def points_with_distance(x,y,d,x_grid,y_grid,tol=1e-12):
    # get all points with distance d from (x,y) on the grid (x_grid,y_grid)
    points = []
    for i in x_grid:
        for j in y_grid:
            distance = np.sqrt((x - i)**2 + (y - j)**2)
            if np.abs(distance - d) < tol:
                points.append((i, j))
    return points