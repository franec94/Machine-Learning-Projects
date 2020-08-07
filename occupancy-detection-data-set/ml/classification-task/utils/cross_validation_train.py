import os
import sys
import time

import numpy as np

from utils.build_models import build_model

from utils.plot_util import plot_val_mae_vs_epochs_regression, plot_smooth_curve_val_mae_vs_epochs_regression
from utils.plot_util import plot_train_vs_val_acc, plot_train_vs_val_loss, get_axes

def k_fold_cv_regression(train_data, train_targets, n_features, num_epochs = 10, k = 4, batch_size = 1, verbose = 0, show_plot = False):

    all_histories, _ = k_fold_cv_(train_data, train_targets, n_features, n_classes = 0, num_epochs = num_epochs, k = k, batch_size = batch_size, verbose = verbose)
    
    all_mae_histories = [history['val_mean_absolute_error'] for history in all_histories]
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories] for i in num_epochs)
    ]
    if show_plot is True:
        plot_val_mae_vs_epochs_regression(average_mae_history, title = 'Standard Curve')
        plot_smooth_curve_val_mae_vs_epochs_regression(average_mae_history, title = 'Smoothed Curve')
        pass
    pass


def k_fold_cv_classification(train_data, train_targets, n_features, n_classes, num_epochs = 10, k = 4, num_units = 64, activation = 'relu', optimizer = 'rmsprop', batch_size = 1, verbose = 0, show_plot = False, fig_size = (6, 4)):

    all_histories, val_results = k_fold_cv_(
        train_data, train_targets, n_features, n_classes = n_classes, num_epochs = num_epochs, k = k,
        num_units = num_units, activation = activation, optimizer = optimizer,
        batch_size = batch_size, verbose = verbose)
    
    all_acc_histories = [history['accuracy'] for history in all_histories]
    val_all_acc_histories = [history['val_accuracy'] for history in all_histories]

    all_loss_histories = [history['loss'] for history in all_histories]
    val_all_loss_histories = [history['val_loss'] for history in all_histories]

    avg_val_loss = np.mean([val_acc for val_acc in val_results[:][0]])
    avg_val_acc = np.mean([val_acc for val_acc in val_results[:][1]])
    
    print(f"Average Validation Accuracy: {avg_val_acc:.2f}")
    print(f"Average Validation Loss: {avg_val_loss:.2f}")

    if show_plot is True:
        print("Show plots:")
        for ii in range(0, len(all_acc_histories)):
            acc, val_acc = \
                all_acc_histories[ii], val_all_acc_histories[ii]
            loss, val_loss = \
                all_loss_histories[ii], val_all_loss_histories[ii]

            epochs = len(acc)

            print(f"k-fold # {ii}: # epochs = {epochs}")

            assert len(acc) == len(val_acc), f"len(acc)={len(acc)} != len(val_acc)={len(val_acc)}"
            assert len(loss) == len(val_loss), f"len(loss)={len(loss)} != len(val_loss)={len(val_loss)}"
            assert len(acc) == len(loss), f"len(acc)={len(acc)} != len(val_loss)={len(val_loss)}"
            
            _, axes = get_axes(n=2, fig_size = fig_size, grid_shape = (1, 2))
            plot_train_vs_val_acc(acc, val_acc, epochs, title = f'Fold # {ii+1} - Training and validation accuracy', ax=axes[0])
            plot_train_vs_val_loss(loss, val_loss, epochs, title = f'Fold # {ii+1} - Training and validation loss', ax=axes[1])
            pass
        pass
    pass


def k_fold_cv_(train_data, train_targets, n_features, n_classes = 0, num_epochs = 10, k = 4,  num_units = 64, activation = 'relu', optimizer = 'rmsprop', batch_size = 1, verbose = 0):
    
    num_val_samples = len(train_data) // k
    all_histories = list()
    val_results = list()

    t0 = time.time()
    for i in range(k):
        print(f'Processing fold # {i+1}/{k} (epochs = {num_epochs})... ', end='')
        t0_ = time.time()
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
            train_data[(i + 1) * num_val_samples:]],
            axis = 0
        )
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
            train_targets[(i + 1) * num_val_samples:]],
            axis = 0
        )

        model = build_model(n_features = n_features, n_classes = n_classes)
        history = model.fit(partial_train_data, partial_train_targets,
            validation_data = (val_data, val_targets),
            epochs = num_epochs, batch_size = batch_size, verbose = verbose)
        t1_ = time.time()
        print(f"Elapsed time: {t1_-t0_:.2f} sec.")

        result = model.evaluate(val_data, val_targets)
        val_results.append(result)

        all_histories.append(history.history)
        pass
    t1 = time.time()
    print(f"Total time K-fold CV: {t1-t0:.2f} sec")
    return all_histories, val_results
