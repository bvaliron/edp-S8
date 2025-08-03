import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from .fem_utils import local_stiffness, local_mass, local_rhs

class FEMSolver:
    """
    Finite Element Method solver for -Δu + κ²u = f on (0,1)² with P_1 elements.
    
    Parameters:
    -----------
    mesh : Mesh object
        Must have attributes: nodes, elements, is_dirichlet
    kappa : float, default=1.0
        Helmholtz parameter κ
    f_val : float, default=1.0
        Right-hand side value (assumed constant)
    """
    def __init__(self, mesh, kappa=1.0, f_val=1.0):
        # Validation des entrées
        if not hasattr(mesh, 'nodes') or not hasattr(mesh, 'elements'):
            raise ValueError("Mesh must have 'nodes' and 'elements' attributes")
        if len(mesh.nodes) == 0:
            raise ValueError("Mesh has no nodes")
        if len(mesh.elements) == 0:
            raise ValueError("Mesh has no elements")
            
        self.mesh = mesh
        self.kappa = float(kappa)
        self.f_val = float(f_val)
        self.A = None
        self.b = None
        self.u = None

    # --------------------------------------------------------
    def assemble(self):
        n = len(self.mesh.nodes)
        data, rows, cols = [], [], []
        rhs = np.zeros(n)
        for tri in self.mesh.elements:
            coords = self.mesh.nodes[tri]
            Ke = local_stiffness(coords)
            Me = local_mass(coords)
            Re = local_rhs(coords, self.f_val)
            Ae = Ke + (self.kappa**2) * Me
            for a in range(3):
                ia = tri[a]
                rhs[ia] += Re[a]
                for b in range(3):
                    ib = tri[b]
                    rows.append(ia); cols.append(ib); data.append(Ae[a, b])
        
        # Assemblage efficace avec gestion des conditions de Dirichlet
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        fixed = np.where(self.mesh.is_dirichlet)[0]
        
        if fixed.size:
            # Méthode efficace : masquer les lignes/colonnes des nœuds fixés
            mask = np.ones(len(data), dtype=bool)
            for i, (r, c) in enumerate(zip(rows, cols)):
                if r in fixed or c in fixed:
                    mask[i] = False
            
            # Garder seulement les entrées valides
            data_filtered = [data[i] for i in range(len(data)) if mask[i]]
            rows_filtered = [rows[i] for i in range(len(rows)) if mask[i]]
            cols_filtered = [cols[i] for i in range(len(cols)) if mask[i]]
            
            # Ajouter les entrées diagonales pour les nœuds fixés
            for idx in fixed:
                data_filtered.append(1.0)
                rows_filtered.append(idx)
                cols_filtered.append(idx)
            
            A = sp.coo_matrix((data_filtered, (rows_filtered, cols_filtered)), shape=(n, n)).tocsr()
            rhs[fixed] = 0.0
        else:
            A = A.tocsr()
            
        self.A, self.b = A, rhs
        return self

    # --------------------------------------------------------
    def solve(self, use_cg=True, tol=1e-10, maxiter=None):
        if self.A is None:
            self.assemble()
        if use_cg:
            sol, info = spla.cg(self.A, self.b, tol=tol, maxiter=maxiter)
            if info != 0:
                sol = spla.spsolve(self.A, self.b)
        else:
            sol = spla.spsolve(self.A, self.b)
        self.u = sol
        return sol
