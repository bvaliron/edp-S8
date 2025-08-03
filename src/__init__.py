"""
Package de solveur d'éléments finis pour l'équation de Helmholtz
"""

from .solver import FEMSolver
from .mesh_square import SquareMesh
from .mesh_triangle import EquilateralMesh
from .fem_utils import *

__all__ = ['FEMSolver', 'SquareMesh', 'EquilateralMesh']
