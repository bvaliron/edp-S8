
"""Structured square mesh split into two right triangles per cell."""
import numpy as np

def create_square_obstacle(N, obstacle_size, center_i=None, center_j=None):
    """
    Crée un obstacle carré aligné sur la grille du maillage.
    
    Args:
        N (int): Résolution du maillage (N x N cellules)
        obstacle_size (int): Taille de l'obstacle en nombre de cellules
        center_i (int, optional): Position centrale en i (colonne). Si None, centré automatiquement.
        center_j (int, optional): Position centrale en j (ligne). Si None, centré automatiquement.
    
    Returns:
        dict: Dictionnaire de configuration de l'obstacle avec les clés:
            - 'i0', 'j0': indices de départ de l'obstacle dans la grille
            - 'x0', 'y0': coordonnées continues du coin inférieur gauche
            - 'm': taille en nombre de cellules
            - 'size_x', 'size_y': taille en coordonnées continues
    """
    h = 1.0 / N
    
    # Si pas de centre spécifié, centrer l'obstacle
    if center_i is None:
        center_i = N // 2
    if center_j is None:
        center_j = N // 2
    
    # Calculer les indices de départ pour centrer l'obstacle
    i0 = max(1, center_i - obstacle_size // 2)
    j0 = max(1, center_j - obstacle_size // 2)
    
    # S'assurer que l'obstacle ne dépasse pas les bords
    i0 = min(i0, N - obstacle_size)
    j0 = min(j0, N - obstacle_size)
    
    # Convertir en coordonnées continues
    x0 = i0 * h
    y0 = j0 * h
    size_x = obstacle_size * h
    size_y = obstacle_size * h
    
    return {
        'i0': i0,
        'j0': j0,
        'x0': x0,
        'y0': y0,
        'm': obstacle_size,
        'size_x': size_x,
        'size_y': size_y
    }

def create_square_obstacle_at_position(N, obstacle_size, corner_i, corner_j):
    """
    Crée un obstacle carré à une position spécifique de la grille.
    
    Args:
        N (int): Résolution du maillage (N x N cellules)
        obstacle_size (int): Taille de l'obstacle en nombre de cellules
        corner_i (int): Indice i du coin inférieur gauche
        corner_j (int): Indice j du coin inférieur gauche
    
    Returns:
        dict: Dictionnaire de configuration de l'obstacle
    """
    h = 1.0 / N
    
    # Vérifier que l'obstacle ne dépasse pas les limites
    if corner_i < 1 or corner_j < 1:
        raise ValueError("L'obstacle ne peut pas commencer sur le bord du domaine")
    if corner_i + obstacle_size > N or corner_j + obstacle_size > N:
        raise ValueError("L'obstacle dépasse les limites du domaine")
    
    x0 = corner_i * h
    y0 = corner_j * h
    
    return {
        'i0': corner_i,
        'j0': corner_j,
        'x0': x0,
        'y0': y0,
        'm': obstacle_size,
        'size_x': obstacle_size * h,
        'size_y': obstacle_size * h
    }

class SquareMesh:
    def __init__(self, N, obstacle=None):
        self.N = int(N)
        self.h = 1.0 / self.N
        self.obstacle = obstacle
        self.nodes = None
        self.elements = None
        self.is_dirichlet = None
        
        # Si l'obstacle utilise un obstacle aligné sur la grille
        if obstacle and 'i0' in obstacle and 'j0' in obstacle:
            self._setup_grid_aligned_obstacle()

    def _setup_grid_aligned_obstacle(self):
        """Configure un obstacle aligné sur la grille."""
        if self.obstacle is None:
            return
        
        # Extraction des paramètres
        i0, j0, m = self.obstacle['i0'], self.obstacle['j0'], self.obstacle['m']
        
        # Calcul des coordonnées continues
        x0 = i0 * self.h
        y0 = j0 * self.h
        
        # Mise à jour de l'obstacle avec les coordonnées continues
        self.obstacle.update({
            'x0': x0,
            'y0': y0,
            'size_x': m * self.h,
            'size_y': m * self.h
        })

    # ------------------------------------------------------------
    def _is_on_grid_point(self, x, y):
        """Vérifie si un point (x,y) est exactement sur un nœud de la grille."""
        eps = 1e-12  # Tolérance très petite pour les erreurs numériques
        i_exact = x / self.h
        j_exact = y / self.h
        
        # Vérifier si i_exact et j_exact sont des entiers (à la tolérance près)
        i_is_int = abs(i_exact - round(i_exact)) < eps
        j_is_int = abs(j_exact - round(j_exact)) < eps
        
        return i_is_int and j_is_int
    
    def _get_grid_indices(self, x, y):
        """Retourne les indices de grille (i, j) pour un point (x, y)."""
        i = round(x / self.h)
        j = round(y / self.h)
        return i, j

    def _inside_obstacle(self, x, y):
        if self.obstacle is None:
            return False
        
        # Pour un obstacle aligné sur la grille, on peut utiliser les indices
        if 'i0' in self.obstacle and 'j0' in self.obstacle:
            return self._inside_obstacle_grid_aligned(x, y)
        
        # Méthode traditionnelle pour compatibilité
        x0, y0, m, h = self.obstacle['x0'], self.obstacle['y0'], self.obstacle['m'], self.h
        # Inclusion stricte pour l'intérieur de l'obstacle
        return (x0 < x < x0 + m*h) and (y0 < y < y0 + m*h)
    
    def _inside_obstacle_grid_aligned(self, x, y):
        """Version optimisée pour les obstacles alignés sur la grille."""
        if not self._is_on_grid_point(x, y):
            return False
        
        i, j = self._get_grid_indices(x, y)
        i0, j0, m = self.obstacle['i0'], self.obstacle['j0'], self.obstacle['m']
        
        # Inclusion stricte : i0 < i < i0+m et j0 < j < j0+m
        return (i0 < i < i0 + m) and (j0 < j < j0 + m)

    def _on_obst_boundary(self, x, y):
        if self.obstacle is None:
            return False
        
        # Pour un obstacle aligné sur la grille, on peut utiliser les indices
        if 'i0' in self.obstacle and 'j0' in self.obstacle:
            return self._on_obst_boundary_grid_aligned(x, y)
        
        # Méthode traditionnelle pour compatibilité
        x0, y0, m, h = self.obstacle['x0'], self.obstacle['y0'], self.obstacle['m'], self.h
        eps = 1e-8
        
        # Vérifier si on est exactement sur le bord de l'obstacle
        on_left = abs(x - x0) < eps
        on_right = abs(x - (x0 + m*h)) < eps
        on_bottom = abs(y - y0) < eps
        on_top = abs(y - (y0 + m*h)) < eps
        
        # Dans la région de l'obstacle (incluant les bords)
        within_x = x0 - eps <= x <= x0 + m*h + eps
        within_y = y0 - eps <= y <= y0 + m*h + eps
        
        return ((on_left or on_right) and within_y) or ((on_bottom or on_top) and within_x)
    
    def _on_obst_boundary_grid_aligned(self, x, y):
        """Version optimisée pour les obstacles alignés sur la grille."""
        if not self._is_on_grid_point(x, y):
            return False
            
        i, j = self._get_grid_indices(x, y)
        i0, j0, m = self.obstacle['i0'], self.obstacle['j0'], self.obstacle['m']
        
        # Sur le bord de l'obstacle si on est sur le périmètre du rectangle [i0, i0+m] x [j0, j0+m]
        # mais pas à l'intérieur strict
        
        # Dans la boîte englobante (incluant les bords)
        in_x_range = i0 <= i <= i0 + m
        in_y_range = j0 <= j <= j0 + m
        
        if not (in_x_range and in_y_range):
            return False
        
        # Sur le bord si on est sur une des quatre faces
        on_left = (i == i0)
        on_right = (i == i0 + m)
        on_bottom = (j == j0)
        on_top = (j == j0 + m)
        
        return on_left or on_right or on_bottom or on_top

    # ------------------------------------------------------------
    def build(self):
        N, h = self.N, self.h
        coords = []
        dflag = []
        for j in range(N+1):
            for i in range(N+1):
                x, y = i*h, j*h
                if self._inside_obstacle(x, y):
                    dflag.append(None)
                else:
                    on_outer = (i==0 or j==0 or i==N or j==N)
                    dflag.append(on_outer or self._on_obst_boundary(x, y))
                coords.append((x, y))
        coords = np.asarray(coords)
        old2new = np.full(coords.shape[0], -1, dtype=int)
        keep_nodes, keep_dir = [], []
        for idx, (c, f) in enumerate(zip(coords, dflag)):
            if f is None:
                continue
            old2new[idx] = len(keep_nodes)
            keep_nodes.append(c)
            keep_dir.append(f)
        keep_nodes = np.asarray(keep_nodes); keep_dir = np.asarray(keep_dir, bool)

        elems = []
        for j in range(N):
            for i in range(N):
                bl = j*(N+1) + i
                br = bl + 1
                tl = bl + (N+1)
                tr = tl + 1
                for tri in ((tl, bl, br), (tl, br, tr)):
                    if any(dflag[k] is None for k in tri):
                        continue
                    elems.append([old2new[k] for k in tri])
        self.nodes, self.elements, self.is_dirichlet = keep_nodes, np.asarray(elems, dtype=int), keep_dir
        return self
