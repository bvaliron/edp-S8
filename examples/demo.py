import argparse, matplotlib.pyplot as plt, matplotlib.tri as mtri, numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mesh_square import SquareMesh, create_square_obstacle
from src.mesh_triangle import EquilateralMesh
from src.solver import FEMSolver

def run_demo(args):
    """Run the FEM demo with specified parameters."""
    try:
        if args.mesh == 'square':
            obstacle = None
            if args.obstacle == 'square':
                obstacle = create_square_obstacle(args.N, obstacle_size=args.m)
                print(f"📐 Obstacle carré aligné créé:")
                print(f"   Position: x0={obstacle['x0']:.3f}, y0={obstacle['y0']:.3f}")
                print(f"   Indices grille: i0={obstacle['i0']}, j0={obstacle['j0']}")
                print(f"   Taille: {obstacle['m']} mailles ({obstacle['size_x']:.3f} unités)")
            mesh = SquareMesh(args.N, obstacle=obstacle).build()
        else:
            obstacle = None
            if args.obstacle == 'disk':
                obstacle = {'type': 'disk', 'r': args.r}
                print(f"🔴 Obstacle circulaire créé: rayon={args.r}")
            mesh = EquilateralMesh(args.N, obstacle=obstacle).build()

        solver = FEMSolver(mesh, kappa=args.kappa, f_val=args.f)
        u = solver.solve()

        if mesh.elements.size == 0:
            print("No elements generated; adjust N or obstacle.")
            return
            
        tri = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
        
        # Création de la figure avec deux sous-graphiques côte à côte
        fig = plt.figure(figsize=(16, 7))
        
        # Graphique 2D (gauche)
        ax1 = plt.subplot(1, 2, 1)
        tcf = ax1.tricontourf(tri, u, levels=30, cmap='viridis')
        ax1.tricontour(tri, u, levels=10, colors='black', alpha=0.4, linewidths=0.5)
        plt.colorbar(tcf, ax=ax1, label='Solution u')
        ax1.set_title(f'Solution 2D (mesh={args.mesh}, N={args.N}, κ={args.kappa})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # Ajout du contour de l'obstacle si présent
        if args.obstacle == 'square' and obstacle is not None:
            import matplotlib.patches as patches
            x0, y0, m = obstacle['x0'], obstacle['y0'], obstacle['m']
            size = m / args.N
            rect = patches.Rectangle((x0, y0), size, size, 
                                   linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            ax1.add_patch(rect)
        elif args.obstacle == 'disk' and obstacle is not None:
            import matplotlib.patches as patches
            circle = patches.Circle([0.5, 0.5], args.r, linewidth=2, 
                                  edgecolor='red', facecolor='none', alpha=0.8)
            ax1.add_patch(circle)
        
        # Graphique 3D (droite)
        ax2 = plt.subplot(1, 2, 2, projection='3d')
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        surf = ax2.plot_trisurf(x, y, u, triangles=mesh.elements, cmap='viridis', alpha=0.8)
        ax2.set_title(f'Solution 3D (max={np.max(u):.4f})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u(x,y)')
        
        # Ajustement de l'angle de vue
        ax2.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        # Affichage de statistiques
        print(f"\n📊 Mesh statistics:")
        print(f"  - Nodes: {len(mesh.nodes)}")
        print(f"  - Elements: {len(mesh.elements)}")
        print(f"  - Dirichlet nodes: {np.sum(mesh.is_dirichlet)}")
        print(f"\n📈 Solution statistics:")
        print(f"  - Min: {np.min(u):.6f}")
        print(f"  - Max: {np.max(u):.6f}")
        print(f"  - Mean: {np.mean(u):.6f}")
        
        if args.obstacle == 'square' and obstacle is not None:
            print(f"\n✅ Obstacle carré parfaitement aligné sur la grille {args.N}×{args.N}")
            print(f"   📍 Visualisation: contour rouge sur les graphiques 2D et 3D")
        elif args.obstacle == 'disk' and obstacle is not None:
            print(f"\n✅ Obstacle circulaire créé avec rayon {args.r}")
            print(f"   📍 Visualisation: contour rouge sur les graphiques 2D et 3D")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve -Δu+κ²u=f on (0,1)² with P_1 FEM.')
    parser.add_argument('--mesh', choices=['square', 'tri'], default='square')
    parser.add_argument('-N', type=int, default=20, help='Resolution parameter')
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--f', type=float, default=1.0)
    parser.add_argument('--obstacle', choices=['none','square','disk'], default='none')
    parser.add_argument('--m', type=int, default=5, help='Obstacle size (square mesh)')
    parser.add_argument('--r', type=float, default=0.25, help='Radius for disk obstacle (tri mesh)')
    args = parser.parse_args()
    run_demo(args)

# Exemples de commandes :
# python demo.py --mesh square -N 40 --obstacle square --m 8   # Obstacle carré de taille 8 mailles
# python demo.py --mesh tri -N 20 --obstacle disk --r 0.2      # Obstacle circulaire rayon 0.2
