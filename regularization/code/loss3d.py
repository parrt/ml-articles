import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

def loss(b0, b1,
         a = 1,
         b = 1,
         c = 0,     # axis stretch
         cx = -10,  # shift center x location
         cy = 5):   # shift center y
    eqn = f"{a:.2f}(b0 - {cx:.2f})^2 + {b:.2f}(b1 - {cy:.2f})^2 + {c:.2f} (b0-{cx:.2f}) (b1-{cy:.2f}) + 100"
    print(eqn)
    return a * (b0 - cx) ** 2 + b * (b1 - cy) ** 2 + c * (b0 - cx) * (b1 - cy) + 100


lmbda = 3
w,h = 15,15
beta0 = np.linspace(-w, w, 100)
beta1 = np.linspace(-h, h, 100)
B0, B1 = np.meshgrid(beta0, beta1)

cx = 7
cy = 1
Z = loss(B0, B1, a=1, b=1.2, c=1, cx=cx, cy=cy)
# Z = np.where(Z<600,Z,np.NAN)

# Create a figure and a 3D Axes
fig = plt.figure(figsize=(4.2,3.1))
ax = fig.add_subplot(111, projection='3d')
vmax = 800
ax.plot_surface(B0, B1, Z, alpha=0.7, cmap='coolwarm', vmax=vmax)
ax.set_xlabel("$\\beta_1$", labelpad=0)
ax.set_ylabel("$\\beta_2$", labelpad=0)
ax.set_zlim(0,500)
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.set_xlim(-w,w)
ax.set_ylim(-h,h)
safe = Circle(xy=(0,0), radius=lmbda, color='grey')
ax.add_patch(safe)
art3d.pathpatch_2d_to_3d(safe, z=0)

ax.plot([cx], [cy], marker='x', markersize=10, color='black')

contr = ax.contour(B0, B1, Z, levels=50, linewidths=.5, cmap='coolwarm',
                   zdir='z', offset=0, vmax=vmax)

ax.view_init(elev=39, azim=-106)
plt.tight_layout()
print(f"../images/reg3D.svg")
plt.savefig(f"../images/reg3D.svg", bbox_inches=0, pad_inches=0)
plt.show()