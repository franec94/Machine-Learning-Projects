import time
import warnings

import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def get_axes(n, figsize, grid_shape=None):

    if grid_shape is not None:
        axes = list()
        fig = plt.figure(figsize=figsize)
        nrows_tmp = n // 2 if n % 2 == 0 else n // 2 + 1
        nrows, ncols = grid_shape
        assert (nrows * ncols) == (nrows_tmp * 2), "grid shape is wrong"
        for _ in range(n):
            axes.append(fig.add_subplot(nrows, ncols, _+1))
            pass
    else:
        axes = [None] * n
        return None, axes
    return fig, axes


def show_violinplot_df(df, x_y_pairs, hue = None):

    for _, (x, y) in enumerate(x_y_pairs):
        sns.catplot(x=x, y=y, hue=hue,
            kind="violin", split=True, data=df)
    plt.show()
    pass


def show_boxplot_df(df, x_y_pairs, hue = None):

    for _, (x, y) in enumerate(x_y_pairs):
        sns.catplot(x=x, y=y, hue=hue,
            kind="box", data=df)
    plt.show()
    pass


def show_barplot_df(df, x_y_pairs, hue = None):

    for _, (x, y) in enumerate(x_y_pairs):
        sns.catplot(x=x, y=y, hue=hue,
            kind="bar", data=df)
    plt.show()
    pass


def show_swarmplot_df(df, x_y_pairs, hue = None):

    for _, (x, y) in enumerate(x_y_pairs):
        sns.catplot(x=x, y=y, hue=hue,
            kind="swarm", data=df)
    plt.show()
    pass


def show_joinplot_df(df, x_y_pairs, kind = None):
    for _, (x, y) in enumerate(x_y_pairs):
        sns.jointplot(x=x, y=y, data=df, kind=kind);
    plt.show()
    pass


def show_joinplot_with_levels_df(df, x_y_pairs, n_levels = 6, zorder = 0):
    for _, (x, y) in enumerate(x_y_pairs):
        sns.jointplot(x, y,
            data=df, color="k").plot_joint(sns.kdeplot, zorder=zorder, n_levels=n_levels)
    plt.show()
    pass
