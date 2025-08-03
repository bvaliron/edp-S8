
"""Shared finite‑element utilities for linear (P₁) triangles."""
import numpy as np

__all__ = ["area_triangle", "local_stiffness", "local_mass", "local_rhs"]

def area_triangle(coords):
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    return 0.5 * abs((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0))

def local_stiffness(coords):
    coords = np.asarray(coords)
    x = coords[:, 0]; y = coords[:, 1]
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    area = area_triangle(coords)
    return (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

def local_mass(coords):
    area = area_triangle(coords)
    return (area / 12.0) * (np.ones((3,3)) + np.eye(3))

def local_rhs(coords, f_val):
    area = area_triangle(coords)
    return np.full(3, f_val * area / 3.0)
