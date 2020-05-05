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


def select_parameters(lmbda, reg, force_symmetric_loss, force_one_nonpredictive):
    while True:
        a = np.random.random() * 10
        b = np.random.random() * 10
        c = np.random.random() * 4 - 1.5
        if force_symmetric_loss:
            b = a # make symmetric
            c = 0
        elif force_one_nonpredictive:
            if np.random.random() > 0.5:
                a = np.random.random() * 15 - 5
                b = .1
            else:
                b = np.random.random() * 15 - 5
                a = .1
            c = 0

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


def plot_loss(boundary, reg,
              boundary_color='#2D435D',
              boundary_dot_color='#E32CA6',
              force_symmetric_loss=False, force_one_nonpredictive=False,
              show_contours=True, contour_levels=50, show_loss_eqn=False,
              show_min_loss=True):
    Z, a, b, c, x, y = \
        select_parameters(lmbda, reg,
                          force_symmetric_loss=force_symmetric_loss,
                          force_one_nonpredictive=force_one_nonpredictive)
    x, y = 5, 5

    eqn = f"{a:.2f}(b0 - {x:.2f})^2 + {b:.2f}(b1 - {y:.2f})^2 + {c:.2f} b0 b1"

    fig,ax = plt.subplots(1,1,figsize=(3.8,3.8))
    if show_loss_eqn:
        ax.set_title(eqn, fontsize=10)
    ax.set_xlabel("x", fontsize=12, labelpad=0)
    ax.set_ylabel("y", fontsize=12, labelpad=-10)
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_yticks([-10,-5,0,5,10])
    ax.set_xlabel(r"$\beta_1$", fontsize=12)
    ax.set_ylabel(r"$\beta_2$", fontsize=12)

    shape = ""
    if force_symmetric_loss:
        shape = "symmetric "
    elif force_one_nonpredictive:
        shape = "orthogonal "
    ax.set_title(f"{reg} constraint w/{shape}loss function", fontsize=11)

    if show_contours:
        ax.contour(B0, B1, Z, levels=contour_levels, linewidths=1.0, cmap='coolwarm')
    else:
        ax.contourf(B0, B1, Z, levels=contour_levels, cmap='coolwarm')

    # Draw axes
    ax.plot([-w,+w],[0,0], '-', c='k', lw=.5)
    ax.plot([0, 0],[-h,h], '-', c='k', lw=.5)

    # Draw boundary
    if boundary is not None:
        ax.plot(boundary[:,0], boundary[:,1], '-', lw=1.5, c=boundary_color)

    # Draw center of loss func
    if show_min_loss:
        ax.scatter([x],[y], s=90, c='k')

    # Draw point on boundary
    eqn = f"{a:.2f}(b0 - {x:.2f})^2 + {b:.2f}(b1 - {y:.2f})^2 + {c:.2f} (b0-{x:.2f}) (b1-{y:.2f})"
    # print(eqn)
    if boundary is not None:
        losses = [loss(*edgeloc, a=a, b=b, c=c, cx=x, cy=y) for edgeloc in boundary]
        minloss_idx = np.argmin(losses)
        coeff = boundary[minloss_idx]
        ax.scatter([coeff[0]], [coeff[1]], s=90, c=boundary_dot_color)

        if force_symmetric_loss:
            if reg=='l2':
                ax.plot([x,0],[y,0], ':', c='k')
            else:
                ax.plot([x,coeff[0]],[y,coeff[1]], ':', c='k')

    plt.tight_layout()


n_trials = 4
contour_levels=50

def show_example(reg, force_symmetric_loss=False, force_one_nonpredictive=False):
    if reg == 'l1':
        boundary = diamond(lmbda=lmbda, n=100)
    else:
        boundary = circle(lmbda=lmbda, n=100)
    for i in range(n_trials):
        plot_loss(boundary=boundary, reg=reg,
                  force_symmetric_loss=force_symmetric_loss,
                  force_one_nonpredictive=force_one_nonpredictive,
                  contour_levels=contour_levels)

        shape_fname = ""
        if force_symmetric_loss:
            shape_fname = "symmetric-"
        elif force_one_nonpredictive:
            shape_fname = "orthogonal-"
        print(f"../images/{reg}-{shape_fname}{i}.svg")
        plt.tight_layout()
        plt.savefig(f"../images/{reg}-{shape_fname}{i}.svg", bbox_inches=0, pad_inches=0)
        plt.show()


def just_contour():
    contour_levels = 400
    cx, cy = 4, .8
    beta0 = np.linspace(-w, w, 300)
    beta1 = np.linspace(-h, h, 300)
    B0, B1 = np.meshgrid(beta0, beta1)
    Z = loss(B0, B1, a=9, b=1, c=4.5, cx=cx, cy=cy)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-2,8)
    ax.set_ylim(-2,8)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel(r"$\beta_1$", fontsize=12)
    # ax.set_ylabel(r"$\beta_2$", fontsize=12)

    ax.contour(B0, B1, Z, levels=contour_levels, linewidths=.8, cmap='coolwarm', vmax=380)

    # Draw axes
    ax.plot([-w, +w], [0, 0], '-', c='k', lw=.5)
    ax.plot([0, 0], [-h, h], '-', c='k', lw=.5)

    lmbda = 2.58
    boundary_color = '#2D435D'
    boundary = diamond(lmbda=lmbda, n=200)
    ax.plot(boundary[:, 0], boundary[:, 1], '-', lw=.5, c=boundary_color)
    boundary = circle(lmbda=lmbda, n=200)
    ax.plot(boundary[:, 0], boundary[:, 1], '-', lw=.5, c=boundary_color)

    # Draw center of loss func
    ax.scatter([cx], [cy], s=20, c='k')

    print(f"../images/contour.png")
    plt.tight_layout()
    plt.savefig(f"../images/contour.png", bbox_inches=0, pad_inches=0, dpi=250)
    plt.show()


just_contour()

# np.random.seed(5) # get reproducible sequence
# show_example(reg='l1')
# np.random.seed(9)
# show_example(reg='l2')
#
# np.random.seed(6)
# show_example(reg='l1', force_symmetric_loss=True)
# np.random.seed(7)
# show_example(reg='l2', force_symmetric_loss=True)
#
# np.random.seed(5)
# show_example(reg='l1', force_one_nonpredictive=True)
# np.random.seed(5)
# show_example(reg='l2', force_one_nonpredictive=True)