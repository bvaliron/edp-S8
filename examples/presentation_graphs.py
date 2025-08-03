"""
Script de génération de graphiques pour la présentation du projet
Affiche différents maillages avec et sans obstacles pour diverses valeurs des paramètres
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mesh_square import SquareMesh, create_square_obstacle
from src.mesh_triangle import EquilateralMesh
from src.solver import FEMSolver

def plot_mesh_only(mesh, title, ax):
    """Affiche uniquement le maillage sans solution"""
    if mesh.elements.size == 0:
        ax.text(0.5, 0.5, 'Pas d\'éléments\ngénérés', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    tri = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    ax.triplot(tri, 'k-', linewidth=0.5, alpha=0.7)
    
    # Séparer les nœuds de Dirichlet (bleus) des nœuds intérieurs (rouges)
    if hasattr(mesh, 'is_dirichlet'):
        dirichlet_nodes = mesh.is_dirichlet
        interior_nodes = ~dirichlet_nodes
        
        # Nœuds intérieurs en rouge
        if np.any(interior_nodes):
            ax.scatter(mesh.nodes[interior_nodes,0], mesh.nodes[interior_nodes,1], 
                      c='red', s=8, alpha=0.6, label='Nœuds intérieurs')
        
        # Nœuds de Dirichlet en bleu
        if np.any(dirichlet_nodes):
            ax.scatter(mesh.nodes[dirichlet_nodes,0], mesh.nodes[dirichlet_nodes,1], 
                      c='blue', s=12, alpha=0.8, label='Nœuds Dirichlet (u=0)')
        
        # Ajouter une légende si on a les deux types
        if np.any(interior_nodes) and np.any(dirichlet_nodes):
            ax.legend(fontsize=8, loc='upper right')
    else:
        # Si pas d'information Dirichlet, afficher tous en rouge
        ax.scatter(mesh.nodes[:,0], mesh.nodes[:,1], c='red', s=8, alpha=0.6)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def plot_solution(mesh, u, title, ax):
    """Affiche la solution sur le maillage"""
    if mesh.elements.size == 0:
        ax.text(0.5, 0.5, 'Pas d\'éléments\ngénérés', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    tri = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    tcf = ax.tricontourf(tri, u, levels=20, cmap='viridis')
    plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title)

def create_mesh_comparison():
    """Compare différents types de maillages"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison des maillages - Sans obstacles', fontsize=16, fontweight='bold')
    
    # Maillages carrés
    N_values = [10, 20, 30]
    for i, N in enumerate(N_values):
        mesh = SquareMesh(N).build()
        plot_mesh_only(mesh, f'Maillage carré N={N}\n({len(mesh.nodes)} nœuds, {len(mesh.elements)} éléments)', axes[0, i])
    
    # Maillages triangulaires
    for i, N in enumerate(N_values):
        mesh = EquilateralMesh(N).build()
        plot_mesh_only(mesh, f'Maillage triangulaire N={N}\n({len(mesh.nodes)} nœuds, {len(mesh.elements)} éléments)', axes[1, i])
    
    plt.tight_layout()
    plt.savefig('images/maillages_comparaison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_obstacles_comparison():
    """Compare maillages avec différents obstacles"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Maillages avec obstacles', fontsize=16, fontweight='bold')
    
    N = 25  # Résolution fixe pour la comparaison
    
    # Première ligne : maillages carrés
    # Sans obstacle
    mesh = SquareMesh(N).build()
    plot_mesh_only(mesh, f'Carré sans obstacle\n({len(mesh.nodes)} nœuds)', axes[0, 0])
    
    # Avec obstacles carrés de différentes tailles
    obstacle_sizes = [4, 6, 8]
    for i, m in enumerate(obstacle_sizes):
        obstacle = create_square_obstacle(N, obstacle_size=m)
        mesh = SquareMesh(N, obstacle=obstacle).build()
        plot_mesh_only(mesh, f'Obstacle carré m={m}\n({len(mesh.nodes)} nœuds)', axes[0, i+1])
    
    # Deuxième ligne : maillages triangulaires avec disques
    # Sans obstacle
    mesh = EquilateralMesh(N).build()
    plot_mesh_only(mesh, f'Triangulaire sans obstacle\n({len(mesh.nodes)} nœuds)', axes[1, 0])
    
    # Avec obstacles disques de différents rayons
    disk_radii = [0.15, 0.25, 0.35]
    for i, r in enumerate(disk_radii):
        obstacle = {'type': 'disk', 'r': r}
        mesh = EquilateralMesh(N, obstacle=obstacle).build()
        plot_mesh_only(mesh, f'Obstacle disque r={r}\n({len(mesh.nodes)} nœuds)', axes[1, i+1])
    
    plt.tight_layout()
    plt.savefig('images/maillages_obstacles.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solutions_showcase():
    """Affiche quelques solutions typiques"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Solutions du problème de Helmholtz', fontsize=16, fontweight='bold')
    
    # Paramètres de solutions
    configs = [
        # (mesh_type, N, obstacle, kappa, title)
        ('square', 30, None, 5.0, 'Carré sans obstacle\nκ=5'),
        ('square', 25, create_square_obstacle(25, obstacle_size=6), 3.0, 'Carré avec obstacle\nκ=3'),
            ('square', 30, None, 10.0, 'Carré sans obstacle\nκ=10'),
        ('tri', 30, None, 5.0, 'Triangulaire sans obstacle\nκ=5'),
        ('tri', 25, {'type': 'disk', 'r': 0.2}, 3.0, 'Triangulaire avec disque\nκ=3'),
        ('tri', 30, {'type': 'disk', 'r': 0.3}, 8.0, 'Triangulaire avec disque\nκ=8'),
    ]
    
    for idx, (mesh_type, N, obstacle, kappa, title) in enumerate(configs):
        row, col = divmod(idx, 3)
        
        try:
            if mesh_type == 'square':
                mesh = SquareMesh(N, obstacle=obstacle).build()
            else:
                mesh = EquilateralMesh(N, obstacle=obstacle).build()
            
            if mesh.elements.size > 0:
                solver = FEMSolver(mesh, kappa=kappa, f_val=1.0)
                u = solver.solve()
                plot_solution(mesh, u, title, axes[row, col])
            else:
                axes[row, col].text(0.5, 0.5, 'Échec du maillage', ha='center', va='center', 
                                  transform=axes[row, col].transAxes, fontsize=12)
                axes[row, col].set_title(title)
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Erreur:\n{str(e)[:50]}...', ha='center', va='center', 
                              transform=axes[row, col].transAxes, fontsize=10)
            axes[row, col].set_title(title)
    
    plt.tight_layout()
    plt.savefig('images/solutions_showcase.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_statistics_plot():
    """Graphique des statistiques de maillage"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Statistiques des maillages', fontsize=16, fontweight='bold')
    
    N_range = range(5, 41, 5)
    
    # Collecte des données
    square_nodes = []
    square_elements = []
    tri_nodes = []
    tri_elements = []
    
    for N in N_range:
        # Maillage carré
        mesh_sq = SquareMesh(N).build()
        square_nodes.append(len(mesh_sq.nodes))
        square_elements.append(len(mesh_sq.elements))
        
        # Maillage triangulaire
        mesh_tri = EquilateralMesh(N).build()
        tri_nodes.append(len(mesh_tri.nodes))
        tri_elements.append(len(mesh_tri.elements))
    
    # Graphique nombre de nœuds
    ax1.plot(N_range, square_nodes, 'bs-', label='Carré', linewidth=2, markersize=8)
    ax1.plot(N_range, tri_nodes, 'ro-', label='Triangulaire', linewidth=2, markersize=8)
    ax1.set_xlabel('Paramètre N')
    ax1.set_ylabel('Nombre de nœuds')
    ax1.set_title('Nombre de nœuds vs N')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique nombre d'éléments
    ax2.plot(N_range, square_elements, 'bs-', label='Carré', linewidth=2, markersize=8)
    ax2.plot(N_range, tri_elements, 'ro-', label='Triangulaire', linewidth=2, markersize=8)
    ax2.set_xlabel('Paramètre N')
    ax2.set_ylabel('Nombre d\'éléments')
    ax2.set_title('Nombre d\'éléments vs N')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/statistics_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Fonction principale pour générer tous les graphiques"""
    print("Génération des graphiques de présentation...")
    print("="*50)
    
    print("1. Comparaison des maillages sans obstacles...")
    create_mesh_comparison()
    
    print("2. Maillages avec obstacles...")
    create_obstacles_comparison()
    
    print("3. Solutions typiques...")
    create_solutions_showcase()
    
    print("4. Statistiques des maillages...")
    create_statistics_plot()
    
    print("="*50)
    print("Tous les graphiques ont été générés et sauvegardés !")
    print("Fichiers créés :")
    print("- maillages_comparaison.png")
    print("- maillages_obstacles.png") 
    print("- solutions_showcase.png")
    print("- statistics_plot.png")

if __name__ == "__main__":
    main()
