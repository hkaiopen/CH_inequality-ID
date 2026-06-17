"""
Visualize the virtual sphere (radius R=2, curvature K=1/4)
and mark the two projections that give 0.25 and 0.207.

Features:
- Draw a sphere with radius 2.
- Shade a spherical cap whose area corresponds to the 0.25 contribution.
- Draw a geodesic arc (great circle) whose length corresponds to the 0.207 contribution.
- Add labels to explain the geometry.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

# ----------------------------------------------------------------------
# Helper to draw a 3D arrow (for annotations)
# ----------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

# ----------------------------------------------------------------------
# Create the figure
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Draw the sphere (radius 2)
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x = 2 * np.outer(np.cos(u), np.sin(v))
y = 2 * np.outer(np.sin(u), np.sin(v))
z = 2 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot semi-transparent sphere
ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='none')

# Draw wireframe to show curvature
ax.plot_wireframe(x, y, z, color='gray', linewidth=0.3, alpha=0.5)

# 2. Mark the "area projection" – a spherical cap (shaded region)
# Define a cap: latitude angle from θ = 0 (north pole) to θ = π/3 (60°)
cap_theta = np.pi / 3   # 60° from north pole (arbitrary, but illustrative)
u_cap = np.linspace(0, 2*np.pi, 30)
v_cap = np.linspace(0, cap_theta, 20)
x_cap = 2 * np.outer(np.cos(u_cap), np.sin(v_cap))
y_cap = 2 * np.outer(np.sin(u_cap), np.sin(v_cap))
z_cap = 2 * np.outer(np.ones(np.size(u_cap)), np.cos(v_cap))
ax.plot_surface(x_cap, y_cap, z_cap, color='red', alpha=0.4, label='Area projection (0.25)')

# 3. Mark the "geodesic arc" – a great circle arc from north pole to a point on equator
# Draw a quarter of a great circle from the north pole to (2,0,0)
phi = np.linspace(0, np.pi/2, 30)
arc_x = 2 * np.sin(phi)   # equatorial direction
arc_y = 0 * np.sin(phi)
arc_z = 2 * np.cos(phi)
ax.plot(arc_x, arc_y, arc_z, color='green', linewidth=4, label='Geodesic arc (0.207)')

# 4. Add labels and annotations
# Label the cap area
ax.text(0, 0, 2.2, 'Area projection\n(0.25)', color='red', ha='center', va='bottom', fontsize=12)
# Label the arc
mid_idx = len(phi)//2
ax.text(arc_x[mid_idx]+0.3, arc_y[mid_idx]-0.2, arc_z[mid_idx]+0.2,
        'Geodesic arc\n(0.207)', color='green', fontsize=12)

# Add a note about curvature
ax.text(0, 0, -2.5, 'Curvature K = 1/4, Radius R = 2', color='black', ha='center', fontsize=12)

# 5. Set axes limits and labels
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-2.5, 2.5])
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.set_zlabel('Z', fontsize=10)
ax.set_title('Virtual Space Sphere (SU(2) quotient)\nBoth 0.25 and 0.207 originate from this geometry', fontsize=14)

# Add legend (custom handles)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='red', alpha=0.4, label='Area projection → 0.25 (I3322)'),
    Line2D([0], [0], color='green', linewidth=4, label='Geodesic arc → 0.207 (CH)'),
    Patch(facecolor='lightblue', alpha=0.2, label='Sphere (R=2)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('virtual_sphere_geometry.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved as 'virtual_sphere_geometry.png'")
