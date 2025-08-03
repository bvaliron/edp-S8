"""
Comparaison des solutions avec et sans obstacle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mesh_square import SquareMesh, create_square_obstacle
from src.mesh_triangle import EquilateralMesh
from src.solver import FEMSolver

def compare_solutions():
    """Compare solutions avec et sans obstacles."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # === MAILLAGE CARRÉ ===
    N = 100
    
    # Sans obstacle
    mesh1 = SquareMesh(N).build()
    solver1 = FEMSolver(mesh1, kappa=1.0, f_val=1.0)
    u1 = solver1.solve()
    tri1 = mtri.Triangulation(mesh1.nodes[:,0], mesh1.nodes[:,1], mesh1.elements)
    
    # Avec obstacle carré
    obstacle = create_square_obstacle(N, obstacle_size=N//10)
    mesh2 = SquareMesh(N, obstacle=obstacle).build()
    solver2 = FEMSolver(mesh2, kappa=1.0, f_val=1.0)
    u2 = solver2.solve()
    tri2 = mtri.Triangulation(mesh2.nodes[:,0], mesh2.nodes[:,1], mesh2.elements)
    
    # Visualisation ligne 1 : maillage carré
    tcf1 = axes[0,0].tricontourf(tri1, u1, levels=20, cmap='viridis')
    axes[0,0].set_title('Carré sans obstacle')
    axes[0,0].axis('equal')
    
    tcf2 = axes[0,1].tricontourf(tri2, u2, levels=20, cmap='viridis')
    axes[0,1].set_title('Carré avec obstacle carré')
    axes[0,1].axis('equal')
    
    # Différence (interpolée sur même grille)
    axes[0,2].triplot(tri2, 'k-', alpha=0.3, linewidth=0.5)
    dirichlet_nodes = mesh2.nodes[mesh2.is_dirichlet]
    axes[0,2].scatter(dirichlet_nodes[:,0], dirichlet_nodes[:,1], c='red', s=5)
    axes[0,2].set_title('Maillage avec obstacle (rouge=Dirichlet)')
    axes[0,2].axis('equal')
    
    # === MAILLAGE TRIANGULAIRE ===
    
    # Sans obstacle
    mesh3 = EquilateralMesh(N).build()
    solver3 = FEMSolver(mesh3, kappa=1.0, f_val=1.0)
    u3 = solver3.solve()
    tri3 = mtri.Triangulation(mesh3.nodes[:,0], mesh3.nodes[:,1], mesh3.elements)
    
    # Avec obstacle disque
    r = 0.25
    mesh4 = EquilateralMesh(N, obstacle={'type': 'disk', 'r': r}).build()
    solver4 = FEMSolver(mesh4, kappa=1.0, f_val=1.0)
    u4 = solver4.solve()
    tri4 = mtri.Triangulation(mesh4.nodes[:,0], mesh4.nodes[:,1], mesh4.elements)
    
    # Visualisation ligne 2 : maillage triangulaire
    tcf3 = axes[1,0].tricontourf(tri3, u3, levels=20, cmap='viridis')
    axes[1,0].set_title('Triangulaire sans obstacle')
    axes[1,0].axis('equal')
    
    tcf4 = axes[1,1].tricontourf(tri4, u4, levels=20, cmap='viridis')
    axes[1,1].set_title('Triangulaire avec obstacle disque')
    axes[1,1].axis('equal')
    
    # Maillage avec cercle théorique
    axes[1,2].triplot(tri4, 'k-', alpha=0.3, linewidth=0.5)
    dirichlet_nodes = mesh4.nodes[mesh4.is_dirichlet]
    axes[1,2].scatter(dirichlet_nodes[:,0], dirichlet_nodes[:,1], c='red', s=5)
    
    # Cercle théorique
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 + r * np.cos(theta)
    circle_y = 0.5 + r * np.sin(theta)
    axes[1,2].plot(circle_x, circle_y, 'g--', linewidth=2)
    axes[1,2].set_title('Maillage avec disque (vert=théorique)')
    axes[1,2].axis('equal')
    
    plt.tight_layout()
    plt.savefig('images/comparison_obstacles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistiques
    print("📊 Statistiques de comparaison:")
    print(f"Carré sans obstacle    : {len(mesh1.nodes)} nœuds, {np.sum(mesh1.is_dirichlet)} Dirichlet")
    print(f"Carré avec obstacle    : {len(mesh2.nodes)} nœuds, {np.sum(mesh2.is_dirichlet)} Dirichlet")
    print(f"Triangul. sans obstacle: {len(mesh3.nodes)} nœuds, {np.sum(mesh3.is_dirichlet)} Dirichlet")
    print(f"Triangul. avec obstacle: {len(mesh4.nodes)} nœuds, {np.sum(mesh4.is_dirichlet)} Dirichlet")

if __name__ == '__main__':
    compare_solutions()
