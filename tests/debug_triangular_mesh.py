"""
Script de diagnostic pour analyser le maillage triangulaire et les nœuds de Dirichlet.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mesh_triangle import EquilateralMesh
import math

def analyze_triangular_mesh():
    """Analyse détaillée du maillage triangulaire."""
    print("🔍 Analyse du maillage triangulaire...")
    
    N = 32
    mesh = EquilateralMesh(N).build()
    
    print(f"N = {N}")
    print(f"h = {mesh.h}")
    print(f"row_h = {math.sqrt(3)/2 * mesh.h}")
    print(f"Nombre théorique de lignes: {int(1 / (math.sqrt(3)/2 * mesh.h)) + 1}")
    print(f"Nombre de nœuds: {len(mesh.nodes)}")
    print(f"Nombre d'éléments triangulaires: {len(mesh.elements)}")
    print(f"Nœuds Dirichlet: {np.sum(mesh.is_dirichlet)}")
    print(f"Nœuds intérieurs: {np.sum(~mesh.is_dirichlet)}")
    
    # Analyser les coordonnées y
    y_coords = mesh.nodes[:, 1]
    
    # Vérifier spécifiquement les bords
    print(f"\nAnalyse des bords:")
    
    # Bord bas (y ≈ 0)
    bottom_mask = np.abs(y_coords) < 1e-12
    print(f"  Bord bas (y ≈ 0): {np.sum(bottom_mask)} nœuds, {np.sum(bottom_mask & mesh.is_dirichlet)} Dirichlet")
    
    # Bord haut (y ≈ 1)
    top_mask = np.abs(y_coords - 1) < 1e-12
    print(f"  Bord haut (y ≈ 1): {np.sum(top_mask)} nœuds, {np.sum(top_mask & mesh.is_dirichlet)} Dirichlet")
    
    # Vérifier si des nœuds sont proches de y = 1
    close_to_top = np.abs(y_coords - 1) < 0.1
    print(f"  Proches du haut (|y-1| < 0.1): {np.sum(close_to_top)} nœuds")
    
    # Y maximum
    y_max = np.max(y_coords)
    print(f"  Y maximum: {y_max:.10f}")
    print(f"  Distance de y_max à 1: {abs(y_max - 1):.10f}")

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Maillage complet
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    tri = mtri.Triangulation(x, y, mesh.elements)
    
    # Dessiner les arêtes du maillage
    ax1.triplot(tri, 'k-', linewidth=0.5, alpha=0.7)
    
    # Dessiner les nœuds par type
    dirichlet_nodes = mesh.is_dirichlet
    interior_nodes = ~dirichlet_nodes
    
    # Nœuds intérieurs en rouge
    if np.any(interior_nodes):
        ax1.scatter(x[interior_nodes], y[interior_nodes], 
                   c='red', s=8, alpha=0.6, label='Nœuds intérieurs')
    
    # Nœuds de Dirichlet en bleu
    if np.any(dirichlet_nodes):
        ax1.scatter(x[dirichlet_nodes], y[dirichlet_nodes], 
                   c='blue', s=12, alpha=0.8, label='Nœuds Dirichlet (u=0)')
    
    ax1.set_title(f'Maillage triangulaire complet (N={N})')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. Focus sur le bord haut
    top_region_mask = y_coords > 0.8
    
    # Créer une triangulation pour la région haute
    top_nodes = np.where(top_region_mask)[0]
    top_elements = []
    for triangle in mesh.elements:
        if np.any(np.isin(triangle, top_nodes)):
            top_elements.append(triangle)
    
    if top_elements:
        top_elements = np.array(top_elements)
        tri_top = mtri.Triangulation(x, y, top_elements)
        
        # Dessiner les arêtes du maillage de la région haute
        ax2.triplot(tri_top, 'k-', linewidth=0.8, alpha=0.7)
    
    # Dessiner les nœuds de la région haute
    top_mask = top_region_mask
    top_dirichlet = top_mask & dirichlet_nodes
    top_interior = top_mask & interior_nodes
    
    if np.any(top_interior):
        ax2.scatter(x[top_interior], y[top_interior], 
                   c='red', s=15, alpha=0.8, label='Nœuds intérieurs')
    
    if np.any(top_dirichlet):
        ax2.scatter(x[top_dirichlet], y[top_dirichlet], 
                   c='blue', s=18, alpha=0.9, label='Nœuds Dirichlet (u=0)')
    
    # Ligne de référence y = 1
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='y = 1', alpha=0.8)
    ax2.set_title('Région haute (y > 0.8) - Détail du maillage')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0.8, 1.02)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/debug_triangular_mesh.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mesh

if __name__ == '__main__':
    mesh = analyze_triangular_mesh()
