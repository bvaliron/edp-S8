
"""Equilateral triangular mesh built row-by-row with horizontal spacing h."""
import numpy as np, math

class EquilateralMesh:
    def __init__(self, N, obstacle=None):
        self.N = int(N)
        self.h = 1.0 / self.N
        self.obstacle = obstacle
        self.nodes = None
        self.elements = None
        self.is_dirichlet = None

    # ------------------------------------------------------------
    def _inside_disk(self, x, y):
        if not (self.obstacle and self.obstacle.get('type') == 'disk'):
            return False
        r = self.obstacle['r']; cx = cy = 0.5
        return (x - cx)**2 + (y - cy)**2 < r**2
    
    def _on_disk_boundary(self, x, y):
        if not (self.obstacle and self.obstacle.get('type') == 'disk'):
            return False
        r = self.obstacle['r']; cx = cy = 0.5
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # Tolérance basée sur la taille du maillage
        tol = self.h * 0.7  # Un peu moins que h pour éviter de prendre trop de nœuds
        # Sur le bord du disque si la distance est proche de r
        return abs(dist - r) < tol and dist >= r * 0.8  # Éviter le centre

    # ------------------------------------------------------------
    def build(self):
        h = self.h; row_h = math.sqrt(3)/2 * h
        pts, dflag = [], []
        j = 0
        while True:
            y = j * row_h
            if y > 1 + 1e-12:
                break
            num = self.N + 1 if j % 2 == 0 else self.N
            for i in range(num):
                x = (i + 0.5*(j%2)) * h
                if x < -1e-12 or x > 1 + 1e-12:
                    continue
                if self._inside_disk(x, y):
                    dflag.append(None)
                else:
                    on_outer = (abs(x) < 1e-12 or abs(x-1) < 1e-12 or
                                abs(y) < 1e-12 or abs(y-1) < 1e-12)
                    on_disk_boundary = self._on_disk_boundary(x, y)
                    dflag.append(on_outer or on_disk_boundary)
                pts.append((x, y))
            j += 1
        pts = np.asarray(pts)
        old2new = np.full(len(pts), -1, int)
        keep, kdir = [], []
        for idx, (c, f) in enumerate(zip(pts, dflag)):
            if f is None: continue
            old2new[idx] = len(keep)
            keep.append(c)
            kdir.append(f)
        keep = np.asarray(keep); kdir = np.asarray(kdir, bool)
        
        # Correction : marquer les nœuds de la ligne la plus haute comme Dirichlet
        if len(keep) > 0:
            y_max = np.max(keep[:, 1])
            # Tolérance basée sur la hauteur de ligne pour identifier la ligne la plus haute
            tol_top = row_h * 0.5  # La moitié de la hauteur entre les lignes
            top_line_mask = np.abs(keep[:, 1] - y_max) < tol_top
            kdir = kdir | top_line_mask  # Marquer ces nœuds comme Dirichlet
        # connectivity via Delaunay
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(keep)
            elems = tri.simplices.copy()
        except ImportError:
            print("Warning: scipy.spatial.Delaunay not available, no elements generated")
            elems = np.empty((0,3), int)
        except Exception as e:
            print(f"Warning: Delaunay triangulation failed: {e}")
            elems = np.empty((0,3), int)
        self.nodes, self.elements, self.is_dirichlet = keep, elems, kdir
        return self
