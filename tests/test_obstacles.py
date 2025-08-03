"""
Script de test pour valider la correction des bugs d'obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mesh_square import SquareMesh, create_square_obstacle
from src.mesh_triangle import EquilateralMesh
from src.solver import FEMSolver

N = 32
def test_square_obstacle():
    """Teste l'obstacle carré avec visualisation du maillage."""
    print("🔍 Test obstacle carré...")
    
    # Ajouter un obstacle carré de taille 6 cellules
    obstacle = create_square_obstacle(N, obstacle_size=6)
    mesh = SquareMesh(N, obstacle=obstacle).build()
    
    # Solveur
    solver = FEMSolver(mesh, kappa=1.0, f_val=1.0)
    u = solver.solve()
    
    # Visualisation
    fig = plt.figure(figsize=(15, 8))
    
    # Triangulation
    tri = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    
    # 1. Maillage (haut gauche)
    ax1 = plt.subplot(2, 2, 1)
    ax1.triplot(tri, 'k-', alpha=0.3, linewidth=0.5)
    
    # Marquer les nœuds de Dirichlet
    dirichlet_nodes = mesh.nodes[mesh.is_dirichlet]
    internal_nodes = mesh.nodes[~mesh.is_dirichlet]
    
    ax1.scatter(dirichlet_nodes[:,0], dirichlet_nodes[:,1], c='red', s=10, label='Dirichlet')
    ax1.scatter(internal_nodes[:,0], internal_nodes[:,1], c='blue', s=10, label='Intérieur')
    ax1.set_title('Maillage et conditions aux limites')
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Solution 2D (bas gauche)
    ax2 = plt.subplot(2, 2, 3)
    tcf = ax2.tricontourf(tri, u, levels=20, cmap='viridis')
    plt.colorbar(tcf, ax=ax2, label='u')
    ax2.set_title('Solution FEM 2D')
    ax2.axis('equal')
    
    # 3. Solution 3D (droite, occupant 2 cellules verticalement)
    ax3 = plt.subplot(1, 2, 2, projection='3d')
    ax3.plot_trisurf(tri, u, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u')
    ax3.set_title('Solution FEM 3D')
    
    plt.tight_layout()
    plt.savefig('images/test_square_obstacle.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Statistiques:")
    print(f"  - Nœuds totaux: {len(mesh.nodes)}")
    print(f"  - Nœuds Dirichlet: {np.sum(mesh.is_dirichlet)}")
    print(f"  - Éléments: {len(mesh.elements)}")
    print()

def test_disk_obstacle():
    """Teste l'obstacle disque avec visualisation du maillage."""
    print("🔍 Test obstacle disque...")
    
    r = 0.25
    mesh = EquilateralMesh(N, obstacle={'type': 'disk', 'r': r}).build()
    
    # Solveur
    solver = FEMSolver(mesh, kappa=1.0, f_val=1.0)
    u = solver.solve()
    
    # Visualisation
    fig = plt.figure(figsize=(15, 8))
    
    # Triangulation
    tri = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    
    # 1. Maillage (haut gauche)
    ax1 = plt.subplot(2, 2, 1)
    ax1.triplot(tri, 'k-', alpha=0.3, linewidth=0.5)
    
    # Marquer les nœuds de Dirichlet
    dirichlet_nodes = mesh.nodes[mesh.is_dirichlet]
    internal_nodes = mesh.nodes[~mesh.is_dirichlet]
    
    ax1.scatter(dirichlet_nodes[:,0], dirichlet_nodes[:,1], c='red', s=10, label='Dirichlet')
    ax1.scatter(internal_nodes[:,0], internal_nodes[:,1], c='blue', s=10, label='Intérieur')
    
    # Dessiner le cercle théorique
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 + r * np.cos(theta)
    circle_y = 0.5 + r * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'g--', linewidth=2, label='Obstacle théorique')
    
    ax1.set_title('Maillage triangulaire et obstacle disque')
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Solution 2D (bas gauche)
    ax2 = plt.subplot(2, 2, 3)
    tcf = ax2.tricontourf(tri, u, levels=20, cmap='viridis')
    plt.colorbar(tcf, ax=ax2, label='u')
    ax2.set_title('Solution FEM 2D')
    ax2.axis('equal')
    
    # 3. Solution 3D (droite, occupant 2 cellules verticalement)
    ax3 = plt.subplot(1, 2, 2, projection='3d')
    ax3.plot_trisurf(tri, u, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u')
    ax3.set_title('Solution FEM 3D')
    
    plt.tight_layout()
    plt.savefig('images/test_disk_obstacle.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Statistiques:")
    print(f"  - Nœuds totaux: {len(mesh.nodes)}")
    print(f"  - Nœuds Dirichlet: {np.sum(mesh.is_dirichlet)}")
    print(f"  - Éléments: {len(mesh.elements)}")
    print(f"  - Rayon obstacle: {r}")
    print()

if __name__ == '__main__':
    print("🧪 Tests de validation des obstacles\n")
    test_square_obstacle()
    test_disk_obstacle()
    print("✅ Tests terminés!")
