import itertools

import numpy as np
import pandas as pd

import os
import sys
import time

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.metrics import confusion_matrix

def plot_val_mae_vs_epochs_regression(average_mae_history, title = None):
    plt.plot(range(1, len(average_mae_history)), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    if title is not None:
        plt.title(title)
    plt.show()
    pass


def plot_smooth_curve_val_mae_vs_epochs_regression(average_mae_history, start_point = 10, end_point = -1, factor = 0.9):
    points = average_mae_history[start_point:end_point]
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
            pass
        pass

    plot_val_mae_vs_epochs_regression(average_mae_history = smoothed_points)
    pass


def get_axes(n, fig_size = (6, 4), grid_shape = None):

    if grid_shape is not None:
        axes = list()
        fig = plt.figure(figsize=fig_size)
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


def plot_via_ax(x, y, ax, title, xlabel, ylabel, labels, default_title):
    ax.plot(x, y[0], 'bo', label = labels[0])
    ax.plot(x, y[1], 'b', label = labels[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(default_title)
        pass
    pass


def standard_plot(x, y, title, xlabel, ylabel, labels, default_title):
    
    plt.plot(x, y[0], 'bo', label = labels[0])
    plt.plot(x, y[1], 'b', label = labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if title is not None:
        plt.title(title)
    else:
        plt.title(default_title)
    pass


def plot_train_vs_val_acc(acc, val_acc, epochs, title = None, ax = None):

    labels = 'Trainining acc,Validation acc'.split(',')
    default_title = 'Training and validation accuracy'
    xlabel, ylabel = 'Epochs', 'Accuracy'
    y = [acc, val_acc]

    epochs_ = list(range(epochs))

    if ax is None:
        _ = plt.figure()
        standard_plot(epochs_, y, title, xlabel, ylabel, labels, default_title)
        plt.show()
    else:
        plot_via_ax(epochs_, y, ax, title, xlabel, ylabel, labels, default_title)
        pass
    pass


def plot_train_vs_val_loss(loss, val_loss, epochs, title = None, ax = None):
    fig = plt.figure()

    labels = 'Trainining loss,Validation loss'.split(',')
    default_title = 'Training and validation loss'
    xlabel, ylabel = 'Epochs', 'Loss'
    y = [loss, val_loss]

    epochs_ = list(range(epochs))
    if ax is None:
        fig = plt.figure()
        standard_plot(epochs_, y, title, xlabel, ylabel, labels, default_title)
        plt.show()
    else:
        plot_via_ax(epochs_, y, ax, title, xlabel, ylabel, labels, default_title)
        pass
    pass


def show_cm(y_test, y_pred, target_names, n_classes = 2, title = 'Confusion Matrix'):
    if n_classes == 2:
        cm = confusion_matrix(y_test, y_pred)
    else:
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    cm_df = pd.DataFrame(cm, columns = target_names, index = target_names)

    res = sns.heatmap(cm_df, annot=True)
    plt.title(title)
    pass


def visualize_pca_reduced_data(kmeans, reduced_data, pc_1 = 0, pc_2 = 1):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, pc_1].min() - 1, reduced_data[:, pc_1].max() + 1
    y_min, y_max = reduced_data[:, pc_2].min() - 1, reduced_data[:, pc_2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

    plt.plot(reduced_data[:, pc_1], reduced_data[:, pc_2], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, pc_1], centroids[:, pc_2],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    pass
