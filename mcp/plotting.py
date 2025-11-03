import matplotlib.pyplot as plt
import numpy as np

def plot_contour_grid(Xi, Yi, field, title="Field", cmap="viridis", levels=50, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(Xi, Yi, field, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax

def plot_quiver(Xi, Yi, Ui, Vi, title="Velocity Field", stride=6, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(Xi[::stride, ::stride], Yi[::stride, ::stride],
              Ui[::stride, ::stride], Vi[::stride, ::stride], scale=3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax

def plot_streamlines(Xi, Yi, Ui, Vi, title="Streamlines", density=1.0, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    speed = np.sqrt(Ui**2 + Vi**2)
    strm = ax.streamplot(Xi, Yi, Ui, Vi, density=density, color=speed, cmap="plasma")
    fig.colorbar(strm.lines, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax

def plot_scatter_coords(coords, values, title="Scatter Plot", cmap="viridis", figsize=(6, 5)):
    x = coords[:, 0]
    y = coords[:, 1]
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x, y, c=values, cmap=cmap, s=8)
    fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax
