import numpy as np
import matplotlib.pyplot as plt
import glob
import os

lmbda = 2
w,h = 10,10
beta0 = np.linspace(-w, w, 100)
beta1 = np.linspace(-h, h, 100)
B0, B1 = np.meshgrid(beta0, beta1)

def diamond(lmbda=1, n=100):
    "get points along diamond at distance lmbda from origin"
    points = []
    x = np.linspace(0, lmbda, num=n // 4)
    points.extend(list(zip(x, -x + lmbda)))

    x = np.linspace(0, lmbda, num=n // 4)
    points.extend(list(zip(x,  x - lmbda)))

    x = np.linspace(-lmbda, 0, num=n // 4)
    points.extend(list(zip(x, -x - lmbda)))

    x = np.linspace(-lmbda, 0, num=n // 4)
    points.extend(list(zip(x,  x + lmbda)))

    return np.array(points)


def circle(lmbda=1, n=100):
    # walk radians around circle, using cos, sin to get x,y
    points = []
    for angle in np.linspace(0,np.pi/2, num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    for angle in np.linspace(np.pi/2,np.pi, num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    for angle in np.linspace(np.pi, np.pi*3/2, num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    for angle in np.linspace(np.pi*3/2, 2*np.pi,num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    return np.array(points)


def loss(b0, b1,
         a = 1,
         b = 1,
         c = 0,     # axis stretch
         cx = -10,  # shift center x location
         cy = 5):   # shift center y
    return a * (b0 - cx) ** 2 + b * (b1 - cy) ** 2 + c * (b0 - cx) * (b1 - cy)


def select_parameters(lmbda, reg):
    while True:
        a = np.random.random() * 10
        b = np.random.random() * 10
        c = np.random.random() * 4 - 1.5

        # get x,y outside of circle radius lmbda
        x, y = 0, 0
        if reg=='l1':
            while np.abs(x) + np.abs(y) <= lmbda:
                x = np.random.random() * 2 * w - w
                y = np.random.random() * 2 * h - h
        else:
            while np.sqrt(x**2 + y**2) <= lmbda:
                x = np.random.random() * 2 * w - w
                y = np.random.random() * 2 * h - h


        Z = loss(B0, B1, a=a, b=b, c=c, cx=x, cy=y)
        loss_at_min = loss(x, y, a=a, b=b, c=c, cx=x, cy=y)
        if (Z >= loss_at_min).all(): # hooray! we didn't make a saddle point
            break # fake repeat-until in python
        # print("loss not min", loss_at_min)

    return Z, a, b, c, x, y


def plot_loss(boundary, reg, show_contours=True, contour_levels=50, show_loss_eqn=False):
    Z, a, b, c, x, y = select_parameters(lmbda, reg)
    eqn = f"{a:.2f}(b0 - {x:.2f})^2 + {b:.2f}(b1 - {y:.2f})^2 + {c:.2f} b0 b1"

    fig,ax = plt.subplots(1,1,figsize=(3.8,3.8))
    if show_loss_eqn:
        ax.set_title(eqn, fontsize=10)
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_yticks([-10,-5,0,5,10])
    ax.set_xlabel(r"$\beta_1$", fontsize=12)
    ax.set_ylabel(r"$\beta_2$", fontsize=12)
    ax.set_title("L2 constraint w/loss function", fontsize=12)


    if show_contours:
        ax.contour(B0, B1, Z, levels=contour_levels, linewidths=1.0, cmap='coolwarm')
    else:
        ax.contourf(B0, B1, Z, levels=contour_levels, cmap='coolwarm')

    # Draw axes
    ax.plot([-w,+w],[0,0], '-', c='k')
    ax.plot([0, 0],[-h,h], '-', c='k')
    ax.plot(boundary[:,0], boundary[:,1], '-', lw=1.5, c='#A22396')

    # Draw center of loss func
    ax.scatter([x],[y], s=90, c='k')

    # Draw point on boundary
    eqn = f"{a:.2f}(b0 - {x:.2f})^2 + {b:.2f}(b1 - {y:.2f})^2 + {c:.2f} (b0-{x:.2f}) (b1-{y:.2f})"
    print(eqn)
    losses = [loss(*edgeloc, a=a, b=b, c=c, cx=x, cy=y) for edgeloc in boundary]
    minloss_idx = np.argmin(losses)
    coeff = boundary[minloss_idx]
    ax.scatter([coeff[0]], [coeff[1]], s=90, c='#D73028')
    plt.tight_layout()
    # plt.show()


np.random.seed(5) # get reproducible sequence
n_trials = 4
reg = 'l2'
contour_levels=50

if reg == 'l1':
    boundary = diamond(lmbda=lmbda, n=100)
else:
    boundary = circle(lmbda=lmbda, n=100)
for i in range(n_trials):
    plot_loss(boundary=boundary, reg=reg, contour_levels=contour_levels)
    print(f"../images/{reg}-frame-{i}.svg")
    plt.savefig(f"../images/{reg}-frame-{i}.svg", bbox_inches=0, pad_inches=0)

