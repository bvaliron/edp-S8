# Solveur Éléments Finis pour une équation d'onde


## Description

Ce projet implémente un solveur par méthode des éléments finis (FEM) pour une équation d'onde :

$$(\mathcal P)\quad
\begin{cases}
\Delta u - \kappa^2 u = f & \text{dans } \mathcal U \\
u = 0 & \text{sur } \partial\mathcal U
\end{cases}$$

sur le domaine carré unitaire $(0,1)^2$ avec conditions aux limites de Dirichlet homogènes. Le solveur utilise des éléments $\mathbb P_1$ (linéaires) et supporte différents types de maillages ainsi que des obstacles.

## Fonctionnalités

- **Maillages supportés** :
  - Maillage carré uniforme (grille régulière divisée en triangles rectangles)
  - Maillage triangulaire équilatéral
- **Obstacles** :
  - Obstacles carrés alignés sur la grille (pour maillages carrés)
  - Obstacles circulaires (pour maillages triangulaires)
- **Solveur robuste** avec gestion d'erreurs et validation des entrées
- **Visualisation** complète des solutions et maillages
- **Interface en ligne de commande** pour les démonstrations

## Structure du Projet

```
edp-S8/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── main.ipynb                   # Notebook principal d'exploration
├── docs/                        # Documentation du projet
│   ├── Rapport projet S8.pdf    # Rapport final
│   └── Sujet Projet EDP S8.pdf
├── src/                         # Code source principal
│   ├── __init__.py
│   ├── solver.py                # Solveur FEM principal
│   ├── mesh_square.py           # Maillages carrés structurés
│   ├── mesh_triangle.py         # Maillages triangulaires équilatéraux
│   └── fem_utils.py             # Utilitaires pour les éléments finis
├── examples/                    # Scripts de démonstration
│   ├── demo.py                  # Démo en ligne de commande
│   ├── compare_obstacles.py     # Comparaison d'obstacles
│   └── presentation_graphs.py   # Génération de graphiques
├── tests/                       # Tests et validation
│   ├── test_obstacles.py        # Tests des obstacles
│   ├── test_refacto_obstacle.py
│   └── debug_triangular_mesh.py
└── images/                      # Images générées par les exemples
    ├── maillages_comparaison.png
    ├── solutions_showcase.png
    └── ...
```

## Installation

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Les dépendances incluent :
- `numpy >= 1.20.0` : Calculs numériques
- `scipy >= 1.7.0` : Algèbre linéaire et solveurs
- `matplotlib >= 3.3.0` : Visualisation

## Utilisation

### 1. Interface en ligne de commande

Le script `examples/demo.py` fournit une interface complète :

```bash
# Maillage carré basique (20×20)
python examples/demo.py --mesh square -N 20

# Maillage triangulaire avec obstacle circulaire
python examples/demo.py --mesh tri -N 15 --obstacle disk --r 0.3

# Maillage carré avec obstacle carré
python examples/demo.py --mesh square -N 25 --obstacle square --m 8

# Paramètres personnalisés
python examples/demo.py --mesh square -N 30 --kappa 5.0 --f 2.0
```

**Options disponibles :**
- `--mesh {square,tri}` : Type de maillage
- `-N` : Résolution du maillage
- `--obstacle {square,disk}` : Type d'obstacle
- `--m` : Taille de l'obstacle carré (en nombre de mailles)
- `--r` : Rayon de l'obstacle circulaire
- `--kappa` : Paramètre κ de l'équation
- `--f` : Valeur du second membre

### 2. Notebook interactif

Ouvrez `main.ipynb` pour une exploration interactive complète avec :
- Comparaisons de maillages à différentes résolutions
- Visualisation des solutions avec et sans obstacles
- Analyse de convergence
- Génération de graphiques personnalisés

### 3. Utilisation explicite dans un script Python
Vous pouvez également utiliser le solveur directement dans vos scripts Python :

```python
import sys
sys.path.append('src')

from src.mesh_square import SquareMesh, create_square_obstacle
from src.solver import FEMSolver

# Créer un maillage avec obstacle
obstacle = create_square_obstacle(N=20, obstacle_size=6)
mesh = SquareMesh(N=20, obstacle=obstacle).build()

# Résoudre l'équation
solver = FEMSolver(mesh, kappa=10.0, f_val=1.0)
solution = solver.solve()

# Visualiser (voir main.ipynb pour les exemples complets)
```

## Exemples d'utilisation

### Comparaison de maillages

```bash
python examples/compare_obstacles.py
```

Génère des comparaisons entre différents types de maillages et d'obstacles.

### Tests de validation

```bash
python tests/test_obstacles.py      # Test des obstacles
python tests/debug_triangular_mesh.py  # Debug maillages triangulaires
```

## Architecture technique

### Classes principales

#### `FEMSolver` (src/solver.py)
- **Rôle** : Solveur principal par éléments finis
- **Fonctionnalités** :
  - Assemblage des matrices de rigidité et de masse
  - Application des conditions aux limites de Dirichlet
  - Résolution du système linéaire (CG ou décomposition LU)
  - Validation robuste des entrées avec messages d'erreur détaillés

#### `SquareMesh` (src/mesh_square.py)
- **Rôle** : Générateur de maillages carrés structurés
- **Fonctionnalités** :
  - Création d'une grille régulière $N\times N$ divisée en triangles rectangles ($2N^2$ éléments)
  - Support d'obstacles carrés avec détection précise intérieur/frontière
  - Numérotation optimisée des nœuds et éléments

#### `EquilateralMesh` (src/mesh_triangle.py)
- **Rôle** : Générateur de maillages triangulaires équilatéraux
- **Fonctionnalités** :
  - Création ligne par ligne avec espacement horizontal h
  - Support d'obstacles circulaires
  - Gestion automatique des conditions aux limites


## Auteur

**Gabriel Abenhaim**  
Projet réalisé dans le cadre d'un projet à CentraleSupélec.
