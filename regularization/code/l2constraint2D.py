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
         cy = 5,    # shift center y
         lmbda=1.0,
         yintercept=100):
    eqn = f"{a:.2f}(b0 - {cx:.2f})^2 + {b:.2f}(b1 - {cy:.2f})^2 + {c:.2f} (b0-{cx:.2f}) (b1-{cy:.2f}) + {yintercept}"
    print(eqn)
    return lmbda * (a * (b0 - cx) ** 2 + b * (b1 - cy) ** 2) + c * (b0 - cx) * (b1 - cy) + yintercept


def plot_constraint(ax):
    lmbda = 3
    beta0 = np.linspace(-20, 20, 100)
    beta1 = np.linspace(-20, 20, 100)
    B0, B1 = np.meshgrid(beta0, beta1)

    vmax = 400
    cx = 0
    cy = 0
    Z = loss(B0, B1, a=1, b=1, c=0, cx=cx, cy=cy, lmbda=7, yintercept=0)
    # Z = np.where(Z<vmax,Z,np.inf)

    cmap = 'copper'
    # Create a figure and a 3D Axes
    # ax.plot_surface(B0, B1, Z, alpha=0.7, cmap=cmap, vmin=0, vmax=vmax)
    contr = ax.contour(B0, B1, Z, levels=22, linewidths=.5, cmap='binary',
                       vmax=vmax)


def plot_loss(ax):
    lmbda = 3
    beta0 = np.linspace(0, 20, 100)
    beta1 = np.linspace(0, -20, 100)
    B0, B1 = np.meshgrid(beta0, beta1)

    cx = 12
    cy = -12
    Z = loss(B0, B1, a=5, b=5, c=0, cx=cx, cy=cy)
    #Z = np.where(Z<500,Z,np.NAN)

    # Create a figure and a 3D Axes
    vmax = 1200
    # ax.plot_surface(B0, B1, Z, alpha=0.7, cmap='coolwarm', vmax=vmax)
    ax.plot([cx], [cy], marker='x', markersize=10, color='black')
    ax.plot([0,cx], [0,cy], ":", c='k', lw=1.0)

    contr = ax.contour(B0, B1, Z, levels=20, linewidths=1, cmap='coolwarm',
                       vmax=vmax)


w, h = 15, 15
fig = plt.figure(figsize=(3.5, 3.1))
ax = fig.add_subplot(111)
ax.set_xlabel("$\\beta_1$", labelpad=0)
ax.set_ylabel("$\\beta_2$", labelpad=0)
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

plot_constraint(ax)
plot_loss(ax)


plt.tight_layout()
print(f"../images/constraint2D.svg")
plt.savefig(f"../images/constraint2D.svg", bbox_inches=0, pad_inches=0, dpi=250)
plt.show()