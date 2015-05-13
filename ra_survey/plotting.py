import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import gauss_2d_anisotropic, uv_to_ed


def scatter_3d(x1, x2, y, xlabel='u', ylabel='v', zlabel='flux', xlim3d=None,
               ylim3d=None, zlim3d=None):
    """
    Do 3d scatter plot.
    :param x1:
        Array-like of first coordinate.
    :param x2:
        Array-like of second coordinate.
    :param y:
        Array-like of values.
    :param xlabel:
    :param ylabel:
    :param zlabel:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c='r', marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # Configure axes
    if xlim3d is not None:
        ax.set_xlim3d(xlim3d[0], xlim3d[1])
    if ylim3d is not None:
        ax.set_ylim3d(ylim3d[0], ylim3d[1])
    if zlim3d is not None:
        ax.set_zlim3d(zlim3d[0], zlim3d[1])
    plt.show()


def scatter_3d_errorbars(x1=None, x2=None, y=None, sy=None, ux1=None, ux2=None,
                         uy=None, xlabel='u, ED', ylabel='v, ED',
                         zlabel='flux, Jy', xlim3d=None, ylim3d=None,
                         zlim3d=None, savefig=None, color_det='r',
                         color_lim='r', lambda_m=None):
    """
    Do 3d plot with errorbars.
    :param x1:
        Array-like of first coordinate.
    :param x2:
        Array-like of second coordinate.
    :param y:
        Array-like of values.
    :param sy:
        Array-like of errors.
    :param xlabel:
    :param ylabel:
    :param zlabel:
    """
    x1 = uv_to_ed(x1, lambda_m=lambda_m)
    x2 = uv_to_ed(x2, lambda_m=lambda_m)
    try:
        ux1 = uv_to_ed(ux1, lambda_m=lambda_m)
        ux2 = uv_to_ed(ux2, lambda_m=lambda_m)
    except TypeError:
        pass
    try:
        x1max = max(abs(x1))
    except:
        x1max = None
    try:
        x2max = max(abs(x2))
    except:
        x2max = None
    try:
        ux1max = max(abs(ux1))
    except:
        ux1max = None
    try:
        ux2max = max(abs(ux2))
    except:
        ux2max = None

    xmax = max(x1max, ux1max, x2max, ux2max)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.hold(True)
    # Plot points
    if x1 is not None and x2 is not None and y is not None:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, y, c=color_det, marker='o')
    # Plot errorbars
    if sy is not None:
        for i in np.arange(0, len(x1)):
            ax.plot([x1[i], x1[i]], [x2[i], x2[i]], [y[i] + sy[i], y[i] - sy[i]],
                    marker="_", color=color_det)
    # Plot upper limits
    if ux1 is not None and ux2 is not None and uy is not None:
        ax.scatter(ux1, ux2, uy, c=color_lim, marker='v')
    # Configure axes
    if xlim3d is not None:
        ax.set_xlim3d(xlim3d[0], xlim3d[1])
    else:
        ax.set_xlim3d(-1.2 * xmax, 1.2 * xmax)
    if ylim3d is not None:
        ax.set_ylim3d(ylim3d[0], ylim3d[1])
    else:
        ax.set_ylim3d(-1.2 * xmax, 1.2 * xmax)
    if zlim3d is not None:
        ax.set_zlim3d(zlim3d[0], zlim3d[1])

    # Label axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

    if savefig:
        plt.savefig(savefig + '.gif')


def gaussian_2d(p, x1range, x2range, lambda_cm, n=100):
    """
    Surface plot of 2d gaussian.
    :param p:
        Parameters. [amplitude, major axis, [e, rotation angle (from x to y)]].
    """
    # Making transparent color map
    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas

    x1 = np.linspace(x1range[0], x1range[1], 100)
    x2 = np.linspace(x2range[0], x2range[1], 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = gauss_2d_anisotropic(p, x1, x2, lambda_cm)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.hold(True)
    surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=theCM,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_all(p, x1, x2, y, sy=None, ux1=None, ux2=None, uy=None, xlabel='u',
             ylabel='v', zlabel='flux', xlim3d=None, ylim3d=None, zlim3d=None,
             n=30, outfile=None, lambda_cm=None):
    """
    Plot model specified by ``p`` and data.
    :param p:
    :param x1:
    :param x2:
    :param y:
    :param sy:
    :param ux1:
    :param ux2:
    :param uy:
    :param xlabel:
    :param ylabel:
    :param zlabel:
    :param xlim3d:
    :param ylim3d:
    :param zlim3d:
    :param n:
    """
    try:
        x1max = max(abs(x1))
    except:
        x1max = None
    try:
        x2max = max(abs(x2))
    except:
        x2max = None
    try:
        ux1max = max(abs(ux1))
    except:
        ux1max = None
    try:
        ux2max = max(abs(ux2))
    except:
        ux2max = None

    xmax = max(x1max, ux1max, x2max, ux2max)
    # Making transparent color map
    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.hold(True)

    # Generating surface plot of model
    x1_ = np.linspace(-1.2 * xmax, 1.2 * xmax, n)
    x2_ = np.linspace(-1.2 * xmax, 1.2 * xmax, n)
    x1_, x2_ = np.meshgrid(x1_, x2_)
    y_ = gauss_2d_anisotropic(p, x1_, x2_, lambda_cm)
    # Plotting model
    surf = ax.plot_surface(x1_, x2_, y_, rstride=1, cstride=1, cmap=theCM,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # If upper limits
    if ux1 is not None and ux2 is not None and uy is not None:
        ax.scatter(ux1, ux2, uy, c='g', marker='v')

    # Plotting data
    # If no errors
    if sy is None:
        ax.scatter(x1, x2, y, c='r', marker='o')
    # If errors
    else:
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        y = np.asarray(y)
        sy = np.asarray(sy)
        # Plot points
        ax.scatter(x1, x2, y, c='r', marker='o')
        # Plot errorbars
        for i in np.arange(0, len(x1)):
            ax.plot([x1[i], x1[i]], [x2[i], x2[i]], [y[i] + sy[i], y[i] -
                                                     sy[i]], marker="_",
                    color='r')
    # Configure axes
    if xlim3d is not None:
        ax.set_xlim3d(xlim3d[0], xlim3d[1])
    else:
        ax.set_xlim3d(-1.2 * xmax, 1.2 * xmax)
    if ylim3d is not None:
        ax.set_ylim3d(ylim3d[0], ylim3d[1])
    else:
        ax.set_ylim3d(-1.2 * xmax, 1.2 * xmax)
    if zlim3d is not None:
        ax.set_zlim3d(zlim3d[0], zlim3d[1])
    # Setting labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.show()
    if outfile:
        fig.savefig(outfile)