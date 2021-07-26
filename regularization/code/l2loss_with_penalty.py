import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import glob
import os
from PIL import Image as PIL_Image

def loss(b0, b1,
         a = 1,
         b = 1,
         c = 0,     # axis stretch
         cx = -10,  # shift center x location
         cy = 5,    # shift center y
         lmbda=1.0,
         yintercept=100):
    eqn = f"{a:.2f}(b0 - {cx:.2f})^2 + {b:.2f}(b1 - {cy:.2f})^2 + {c:.2f} (b0-{cx:.2f}) (b1-{cy:.2f}) + {yintercept}"
    return lmbda * (a * (b0 - cx) ** 2 + b * (b1 - cy) ** 2) + c * (b0 - cx) * (b1 - cy) + yintercept



for f in glob.glob(f'/tmp/constraint3D-frame-*.png'):
    os.remove(f)

# repeat last one a few times to get pause
last_lmbda = 6.0
stepsize = .3
lmbdas = list(np.arange(0, last_lmbda, step=stepsize))
lmbdas = [0]*3 + lmbdas + [last_lmbda]*3
for i,lmbda in enumerate(lmbdas):
    fig = plt.figure(figsize=(4.2, 3.1))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("$\\beta_1$", labelpad=0)
    ax.set_ylabel("$\\beta_2$", labelpad=0)
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.set_zlim(0, 1400)

    cx = 15
    cy = -15

    ax.plot([cx], [cy], marker='x', markersize=10, color='black')
    ax.text(-20,20,800, f"$\lambda={lmbda:.1f}$", fontsize=14)

    beta0 = np.linspace(-30, 30, 300)
    beta1 = np.linspace(-30, 30, 300)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z1 = loss(B0, B1, a=1, b=1, c=0, cx=0, cy=0, lmbda=lmbda, yintercept=0)
    Z2 = loss(B0, B1, a=5, b=5, c=0, cx=cx, cy=cy, yintercept=0)
    Z = Z1 + Z2

    # minx_idx = np.argmin(Z, axis=0)[0]
    # miny_idx = np.argmin(Z, axis=1)[0]
    # minx, miny = beta0[minx_idx], beta1[miny_idx]

    origin = Circle(xy=(0, 0), radius=1, color='k')
    ax.add_patch(origin)
    art3d.pathpatch_2d_to_3d(origin, z=0, zdir="z")

    scale = 1.5
    vmax = 8000
    contr = ax.contour(B0, B1, Z, levels=50, linewidths=.5,
                       cmap='coolwarm',
                       zdir='z', offset=0, vmax=vmax)

    j = lmbda*scale
    b0 = (j, 20-j)
    beta0 = np.linspace(-j, 25-j, 300)
    beta1 = np.linspace(-25+j, j, 300)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z1 = loss(B0, B1, a=1, b=1, c=0, cx=0, cy=0, lmbda=lmbda, yintercept=0)
    Z2 = loss(B0, B1, a=5, b=5, c=0, cx=cx, cy=cy, yintercept=0)
    Z = Z1 + Z2

    vmax = 2700
    ax.plot_surface(B0, B1, Z, alpha=1.0, cmap='coolwarm', vmax=vmax)

    ax.view_init(elev=38, azim=-134)

    # x = lmbda * np.sin(np.pi/4)
    # y = -lmbda * np.cos(np.pi/4)
    # end = Circle(xy=(-minx, -miny), radius=1, color='#A22396')
    # ax.add_patch(end)
    # art3d.pathpatch_2d_to_3d(end, z=0, zdir="z")

    plt.tight_layout()
    print(f"/tmp/constraint3D-frame-{i:02d}.png")
    plt.savefig(f"/tmp/constraint3D-frame-{i:02d}.png", bbox_inches=0, pad_inches=0, dpi=200)
    plt.show()
    plt.close()

images = [PIL_Image.open(image) for image in sorted(glob.glob(f'/tmp/constraint3D-frame-*.png'))]
images += reversed(images)
images[0].save(f'../images/lagrange-animation.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,
               optimize=False,
               loop=0)
print(f"Saved ../images/lagrange-animation.gif")
