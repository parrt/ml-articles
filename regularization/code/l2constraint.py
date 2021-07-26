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
    cmap = 'copper'
    vmax = 300

    beta0 = np.linspace(-10, 15, 100)
    beta1 = np.linspace(-15, 15, 100)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z = loss(B0, B1, a=1, b=1, c=0, cx=0, cy=0, lmbda=6, yintercept=0)

    contr = ax.contour(B0, B1, Z, levels=13, linewidths=.3, cmap='binary',
                       zdir='z', offset=0, vmax=vmax)

    w = 7.8
    beta0 = np.linspace(-w, w, 100)
    beta1 = np.linspace(-w, w, 100)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z = loss(B0, B1, a=1, b=1, c=0, cx=0, cy=0, lmbda=6, yintercept=0)
    ax.plot_surface(B0, B1, Z, alpha=0.9, cmap=cmap, vmin=0, vmax=vmax)


def plot_loss(ax):
    vmax = 750
    beta0 = np.linspace(0, 20, 100)
    beta1 = np.linspace(0, -20, 100)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z = loss(B0, B1, a=5, b=5, c=0, cx=12, cy=-12)
    contr = ax.contour(B0, B1, Z, levels=10, linewidths=1.5, cmap='coolwarm',
                       zdir='z', offset=0, vmax=vmax)

    ax.plot([12], [-12], marker='x', markersize=10, color='black')

    beta0 = np.linspace(3, 20, 100)
    beta1 = np.linspace(-3, -20, 100)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z = loss(B0, B1, a=5, b=5, c=0, cx=12, cy=-12)
    ax.plot_surface(B0, B1, Z, alpha=0.9, cmap='coolwarm', vmax=vmax)


w, h = 15, 15
fig = plt.figure(figsize=(4.2, 3.1))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("$\\beta_1$", labelpad=0)
ax.set_ylabel("$\\beta_2$", labelpad=0)
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.set_xlim(-15, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 500)

plot_constraint(ax)
plot_loss(ax)

ax.view_init(elev=38, azim=-134)

plt.tight_layout()
print(f"../images/constraint3D.svg")
plt.savefig(f"../images/constraint3D.svg", bbox_inches=0, pad_inches=0)
plt.show()