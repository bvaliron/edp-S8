"""
Test du refactoring des obstacles carrés alignés sur la grille.
Compare l'ancienne et la nouvelle approche.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mesh_square import SquareMesh, create_square_obstacle
from src.solver import FEMSolver

def test_refacto_obstacle():
    """Test de comparaison entre ancienne et nouvelle approche."""
    
    print("🧪 Test du refactoring des obstacles carrés")
    
    N = 20
    obstacle_size = 6
    
    # === ANCIENNE APPROCHE ===
    print("🔄 Test avec ancienne approche...")
    
    # Création manuelle comme avant
    m = obstacle_size
    old_obstacle = {
        'x0': 0.5 - m/(2*N), 
        'y0': 0.5 - m/(2*N), 
        'm': m
    }
    
    mesh_old = SquareMesh(N, obstacle=old_obstacle).build()
    solver_old = FEMSolver(mesh_old, kappa=1.0, f_val=1.0)
    u_old = solver_old.solve()
    
    # === NOUVELLE APPROCHE ===
    print("✨ Test avec nouvelle approche (alignée sur la grille)...")
    
    # Création avec fonction utilitaire
    new_obstacle = create_square_obstacle(N, obstacle_size)
    print(f"Obstacle généré: {new_obstacle}")
    
    mesh_new = SquareMesh(N, obstacle=new_obstacle).build()
    solver_new = FEMSolver(mesh_new, kappa=1.0, f_val=1.0)
    u_new = solver_new.solve()
    
    # === COMPARAISON ===
    print("\n📊 Comparaison des statistiques:")
    print(f"Ancienne approche:")
    print(f"  - Nœuds totaux: {len(mesh_old.nodes)}")
    print(f"  - Nœuds Dirichlet: {np.sum(mesh_old.is_dirichlet)}")
    print(f"  - Éléments: {len(mesh_old.elements)}")
    
    print(f"Nouvelle approche:")
    print(f"  - Nœuds totaux: {len(mesh_new.nodes)}")
    print(f"  - Nœuds Dirichlet: {np.sum(mesh_new.is_dirichlet)}")
    print(f"  - Éléments: {len(mesh_new.elements)}")
    
    # === VISUALISATION ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparaison : Ancienne vs Nouvelle approche pour obstacles carrés', fontsize=16)
    
    # Triangulations
    tri_old = mtri.Triangulation(mesh_old.nodes[:,0], mesh_old.nodes[:,1], mesh_old.elements)
    tri_new = mtri.Triangulation(mesh_new.nodes[:,0], mesh_new.nodes[:,1], mesh_new.elements)
    
    # === LIGNE 1: ANCIENNE APPROCHE ===
    
    # 1. Maillage ancien
    axes[0,0].triplot(tri_old, 'k-', alpha=0.3, linewidth=0.5)
    dirichlet_old = mesh_old.nodes[mesh_old.is_dirichlet]
    axes[0,0].scatter(dirichlet_old[:,0], dirichlet_old[:,1], c='red', s=8, alpha=0.7)
    axes[0,0].set_title('Ancienne approche - Maillage\n(rouge = Dirichlet)')
    axes[0,0].axis('equal')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Solution ancienne
    tcf_old = axes[0,1].tricontourf(tri_old, u_old, levels=20, cmap='viridis')
    axes[0,1].set_title('Ancienne approche - Solution')
    axes[0,1].axis('equal')
    plt.colorbar(tcf_old, ax=axes[0,1])
    
    # 3. Zoom sur l'obstacle ancien
    axes[0,2].triplot(tri_old, 'k-', alpha=0.3, linewidth=0.5)
    axes[0,2].scatter(dirichlet_old[:,0], dirichlet_old[:,1], c='red', s=15)
    x0_old, y0_old, m_old = old_obstacle['x0'], old_obstacle['y0'], old_obstacle['m']
    h = 1.0/N
    # Rectangle théorique de l'obstacle
    rect_old = plt.Rectangle((x0_old, y0_old), m_old*h, m_old*h, 
                            fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    axes[0,2].add_patch(rect_old)
    axes[0,2].set_xlim(x0_old - 2*h, x0_old + (m_old + 2)*h)
    axes[0,2].set_ylim(y0_old - 2*h, y0_old + (m_old + 2)*h)
    axes[0,2].set_title('Zoom obstacle ancien\n(bleu = théorique)')
    axes[0,2].axis('equal')
    axes[0,2].grid(True, alpha=0.3)
    
    # === LIGNE 2: NOUVELLE APPROCHE ===
    
    # 1. Maillage nouveau
    axes[1,0].triplot(tri_new, 'k-', alpha=0.3, linewidth=0.5)
    dirichlet_new = mesh_new.nodes[mesh_new.is_dirichlet]
    axes[1,0].scatter(dirichlet_new[:,0], dirichlet_new[:,1], c='red', s=8, alpha=0.7)
    axes[1,0].set_title('Nouvelle approche - Maillage\n(rouge = Dirichlet)')
    axes[1,0].axis('equal')
    axes[1,0].grid(True, alpha=0.3)
    
    # 2. Solution nouvelle
    tcf_new = axes[1,1].tricontourf(tri_new, u_new, levels=20, cmap='viridis')
    axes[1,1].set_title('Nouvelle approche - Solution')
    axes[1,1].axis('equal')
    plt.colorbar(tcf_new, ax=axes[1,1])
    
    # 3. Zoom sur l'obstacle nouveau
    axes[1,2].triplot(tri_new, 'k-', alpha=0.3, linewidth=0.5)
    axes[1,2].scatter(dirichlet_new[:,0], dirichlet_new[:,1], c='red', s=15)
    x0_new, y0_new = new_obstacle['x0'], new_obstacle['y0']
    size_x, size_y = new_obstacle['size_x'], new_obstacle['size_y']
    # Rectangle théorique de l'obstacle
    rect_new = plt.Rectangle((x0_new, y0_new), size_x, size_y, 
                            fill=False, edgecolor='green', linewidth=2, linestyle='--')
    axes[1,2].add_patch(rect_new)
    axes[1,2].set_xlim(x0_new - 2*h, x0_new + size_x + 2*h)
    axes[1,2].set_ylim(y0_new - 2*h, y0_new + size_y + 2*h)
    axes[1,2].set_title('Zoom obstacle nouveau\n(vert = théorique)')
    axes[1,2].axis('equal')
    axes[1,2].grid(True, alpha=0.3)
    
    # Ajouter les points de grille pour visualiser l'alignement
    for i in range(N+1):
        for j in range(N+1):
            x_grid, y_grid = i*h, j*h
            if (x0_new - h <= x_grid <= x0_new + size_x + h and 
                y0_new - h <= y_grid <= y0_new + size_y + h):
                axes[1,2].plot(x_grid, y_grid, 'ko', markersize=3, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('images/test_refacto_obstacle.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === VÉRIFICATION DE LA PRÉCISION ===
    print(f"\n🎯 Vérification de l'alignement sur la grille:")
    
    # Vérifier que tous les nœuds Dirichlet de l'obstacle sont sur des points de grille
    obstacle_dirichlet = mesh_new.nodes[mesh_new.is_dirichlet]
    h = mesh_new.h
    
    perfectly_aligned = 0
    for node in obstacle_dirichlet:
        x, y = node
        i_exact = x / h
        j_exact = y / h
        if abs(i_exact - round(i_exact)) < 1e-12 and abs(j_exact - round(j_exact)) < 1e-12:
            perfectly_aligned += 1
    
    print(f"  - Nœuds Dirichlet parfaitement alignés: {perfectly_aligned}/{len(obstacle_dirichlet)}")
    print(f"  - Pourcentage d'alignement: {100*perfectly_aligned/len(obstacle_dirichlet):.1f}%")
    
    print("\n✅ Test terminé!")

if __name__ == '__main__':
    test_refacto_obstacle()
