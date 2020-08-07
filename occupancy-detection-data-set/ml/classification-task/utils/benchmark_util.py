from utils.libs import *


def view_histogram_perm_scores_via_ax(ax, permutation_scores, score, pvalue, n_classes, title = "Permutation Test"):
    ax.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')
    ylim = ax.get_ylim()
    # BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
    # plt.vlines(score, ylim[0], ylim[1], linestyle='--',
    #          color='g', linewidth=3, label='Classification Score'
    #          ' (pvalue %s)' % pvalue)
    # plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
    #          color='k', linewidth=3, label='Luck')
    ax.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
    ax.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    ax.set_title(title)
    ax.set_ylim(ylim)
    plt.legend()
    ax.set_xlabel('Score')
    pass


def view_histogram_perm_scores_via_plot(permutation_scores, score, pvalue, n_classes, title = "Permutation Test", save_fig = False, fig_name = "permutation_test.png"):
    plt.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')
    ylim = plt.ylim()
    # BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
    # plt.vlines(score, ylim[0], ylim[1], linestyle='--',
    #          color='g', linewidth=3, label='Classification Score'
    #          ' (pvalue %s)' % pvalue)
    # plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
    #          color='k', linewidth=3, label='Luck')
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    plt.title(title)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.show()
    if save_fig is True:
        plt.savefig(fig_name)
    pass


def view_histogram_perm_scores(permutation_scores, score, pvalue, n_classes, ax = None, title = "Permutation Test", save_fig = False, fig_name = "permutation_test.png"):
    if ax is None:
        view_histogram_perm_scores_via_plot(permutation_scores, score, pvalue, n_classes, title = title, save_fig = False, fig_name = "permutation_test.png")
    else:
        view_histogram_perm_scores_via_ax(ax, permutation_scores, score, pvalue, n_classes, title = title)
    pass


def add_plot_banchmark(results):
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()
    pass


def benchmark(clf, X_train, y_train, X_test, y_test, target_names, feature_names, opts, preprocessor = None, save_fig = False, fig_dest = None, fig_size = (15, 10), grid_shape = (2,2)):

    clf_descr = str(clf).split('(')[0]

    n = grid_shape[0] * grid_shape[1]
    _, axes = get_axes(n, fig_size = fig_size, grid_shape = grid_shape)
    pos = 0

    n_classes = len(feature_names)
    clf_cloned = sklearn.clone(clf)
    score, pvalue = None, None

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, y_pred,
                                            target_names=target_names))

    if opts.print_cm:
        if axes is None:
            _ = plt.figure()
            print("confusion matrix:")
        # print(metrics.confusion_matrix(y_test, pred))
        clf_name = str(clf).split('(')[0]
        # show_cm(y_test, y_pred = pred, target_names = target_names, n_classes = 2, title = f'{clf_name}: Confusion Matrix', ax = axes[pos])
        metrics.plot_confusion_matrix(estimator=clf, X=X_test, y_true=y_test, ax = axes[pos], normalize = 'true')
        axes[pos].set_title("Confusion Matrix")
        if axes is None:
            plt.show()
            if save_fig is True:
                plt.savefig(os.path.join(fig_dest, clf_descr, "confusion_matrix.png"))
        pos += 1

    if opts.print_roc_curve:
        if axes is None:
            _ = plt.figure()
            print("roc curve:")
        metrics.plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax = axes[pos])
        axes[pos].set_title("Roc Curve")
        if axes is None:
            plt.show()
            if save_fig is True:
                plt.savefig(os.path.join(fig_dest, clf_descr, "roc_curve.png"))
        pos += 1
        pass

    if  opts.print_precision_recall_curve:
        if axes is None:
            _ = plt.figure()
            print("precision-recall curve:")
        metrics.plot_precision_recall_curve(estimator=clf, X=X_test, y=y_test, ax = axes[pos])
        axes[pos].set_title("Precision-Recall Curve")
        if axes is None:
            plt.show()
            if save_fig is True:
                plt.savefig(os.path.join(fig_dest, clf_descr, "precision-recall.png"))
        pos += 1
        pass

    if opts.print_perm_test:
        print("permutation test:")
        
        cv = StratifiedKFold(2)
        score, permutation_scores, pvalue = permutation_test_score(
            clf_cloned, X_train, y_train, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
        print("Classification score %s (pvalue : %s)" % (score, pvalue))
        view_histogram_perm_scores(permutation_scores, score, pvalue, n_classes, ax = axes[pos], )

    if axes is not None:
        plt.show()
        if save_fig is True:
            plt.savefig(os.path.join(fig_dest, f"{clf_descr}.png"))

    print()
    return clf_descr, score, train_time, test_time, score, pvalue


def wrapper_bechmark(X_train, y_train, X_test, y_test, target_names, feature_names, opts, preprocessor = None, save_fig = False, fig_dest = None):

    results = []
    for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, X_train, y_train, X_test, y_test, target_names, feature_names, opts, save_fig = save_fig, fig_dest = fig_dest))

    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3),
                                       X_train, y_train, X_test, y_test, target_names, feature_names, opts, save_fig = save_fig, fig_dest = fig_dest))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty),
                                           X_train, y_train, X_test, y_test, target_names, feature_names, opts, save_fig = save_fig, fig_dest = fig_dest))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet"),
                                       X_train, y_train, X_test, y_test, target_names, feature_names, opts, save_fig = save_fig, fig_dest = fig_dest))

    # Train NearestCentroid without threshold
    '''
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid(),
        X_train, y_train, X_test, y_test, target_names, feature_names, opts, save_fig = save_fig, fig_dest = fig_dest))
    '''
    
    # Train sparse Naive Bayes classifiers
    """
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01),
        X_train, y_train, X_test, y_test, target_names, feature_names, opts, preprocessor))
    results.append(benchmark(BernoulliNB(alpha=.01),
        X_train, y_train, X_test, y_test, target_names, feature_names, opts, preprocessor))
    results.append(benchmark(ComplementNB(alpha=.1),
        X_train, y_train, X_test, y_test, target_names, feature_names, opts, preprocessor))
    """

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
        ('classification', LinearSVC(penalty="l2"))]),
        X_train, y_train, X_test, y_test, target_names, feature_names, opts, save_fig = save_fig, fig_dest = fig_dest))
    
    return results
