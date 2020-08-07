from utils.libs import *


def add_missing_values(X_full, y_full, missing_rate = 0.75, random_state = 42):

    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_full.shape

    # Add missing values in 75% of the lines, default.
    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=np.bool)
    missing_samples[: n_missing_samples] = True

    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[missing_samples, missing_features] = np.nan
    y_missing = y_full.copy()

    return X_missing, y_missing


def get_scores_for_imputer(imputer, scaler, estimator, X_train, y_train, scoring, cv = 5):
    estimator_ = make_pipeline(imputer, scaler, estimator)
    impute_scores = cross_val_score(estimator_, X_train, y_train,
                                    scoring=scoring,
                                    cv=cv)
    return impute_scores


def get_full_score(estimator, scaler, X_train, y_train, scoring, cv = 5):
    estimator_ = make_pipeline(scaler, estimator)
    full_scores = cross_val_score(estimator, X_train, y_train,
                                  scoring=scoring,
                                  cv=cv)
    return (full_scores.mean(), full_scores.std())


def get_impute_zero_score(estimator, scaler, X_missing, y_missing, scoring, add_indicator = True, fill_value = 0, cv = 5):

    imputer = SimpleImputer(missing_values=np.nan, add_indicator=add_indicator,
                            strategy='constant', fill_value=fill_value)
    zero_impute_scores = get_scores_for_imputer(imputer, scaler, estimator, X_missing, y_missing, scoring, cv = cv)
    return (zero_impute_scores.mean(), zero_impute_scores.std())


def get_impute_knn_score(estimator, scaler, X_missing, y_missing, scoring, add_indicator = True, cv = 5):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=add_indicator)
    knn_impute_scores = get_scores_for_imputer(imputer, scaler, estimator, X_missing, y_missing, scoring, cv = cv)
    return (knn_impute_scores.mean(), knn_impute_scores.std())


def get_impute_mean(estimator, scaler, X_missing, y_missing, scoring, add_indicator = True, cv = 5):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean",
                            add_indicator=add_indicator)
    mean_impute_scores = get_scores_for_imputer(imputer, scaler, estimator, X_missing, y_missing, scoring, cv = cv)
    return mean_impute_scores.mean(), mean_impute_scores.std()


def get_impute_median(estimator, scaler, X_missing, y_missing, scoring, add_indicator = True, cv = 5):
    imputer = SimpleImputer(missing_values=np.nan, strategy="median",
                            add_indicator=add_indicator)
    mean_impute_scores = get_scores_for_imputer(imputer, scaler, estimator, X_missing, y_missing, scoring, cv = cv)
    return mean_impute_scores.mean(), mean_impute_scores.std()


def get_impute_most_freq(estimator, scaler, X_missing, y_missing, scoring, add_indicator = True, cv = 5):
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent",
                            add_indicator=add_indicator)
    mean_impute_scores = get_scores_for_imputer(imputer, scaler, estimator, X_missing, y_missing, scoring, cv = cv)
    return mean_impute_scores.mean(), mean_impute_scores.std()


def get_impute_iterative(estimator, scaler, X_missing, y_missing, scoring, add_indicator = True, n_nearest_features = 5, random_state = 0, sample_posterior = True, cv = 5):
    imputer = IterativeImputer(missing_values=np.nan, add_indicator=add_indicator,
                               random_state=random_state, n_nearest_features=n_nearest_features,
                               sample_posterior=sample_posterior)
    iterative_impute_scores = get_scores_for_imputer(imputer,
                                                     scaler,
                                                     estimator,
                                                     X_missing,
                                                     y_missing,
                                                     scoring, cv=cv)
    return (iterative_impute_scores.mean(), iterative_impute_scores.std())


def plot_different_imputation_techs(x_labels, results_performance, scoring, title = 'Imputation Techniques'):
    n_bars = len(results_performance)
    xval = np.arange(n_bars)

    colors_ = ['r', 'g', 'b', 'orange', 'black', 'brown', 'pink']
    colors = colors_[:n_bars]

    # plot diabetes results
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    for j in xval:
        ax1.barh(j, results_performance[j][0], xerr=results_performance[j][1],
             # color=colors[j], alpha=0.6, align='center')
             alpha=0.6, align='center')

    ax1.set_title(title)
    ax1.set_xlim(left=np.min(results_performance) * 0.9,
             right=np.max(results_performance) * 1.1)
    ax1.set_yticks(xval)
    ax1.set_xlabel(scoring)
    ax1.invert_yaxis()
    ax1.set_yticklabels(x_labels)

    plt.show()

    pass


def benchmark_imputing_missing_values(estimator, scaler, scoring, X_train, y_train, random_state = 42, cv = 5, missing_rate = .75):

    results_performance = []
    x_labels = ['Full data',
            'Zero imputation',
            'Mean Imputation',
            'Median Imputation',
            'Most Frequent Imputation',
            'KNN Imputation',
            'Iterative Imputation']

    X_missing, y_missing = add_missing_values(X_full = X_train, y_full = y_train, missing_rate = missing_rate, random_state = random_state)

    estimator_ = sklearn.clone(estimator)
    res = get_full_score(estimator_, scaler,  X_train = X_train, y_train = y_train, scoring = scoring, cv = cv)
    results_performance.append(res)

    estimator_ = sklearn.clone(estimator)
    res = get_impute_zero_score(estimator_, scaler,  X_missing = X_missing, y_missing = y_missing, scoring = scoring, cv = cv)
    results_performance.append(res)

    estimator_ = sklearn.clone(estimator)
    res = get_impute_mean(estimator_, scaler,  X_missing = X_missing, y_missing = y_missing, scoring = scoring, cv = cv)
    results_performance.append(res)
    
    estimator_ = sklearn.clone(estimator)
    res = get_impute_median(estimator_, scaler,  X_missing = X_missing, y_missing = y_missing, scoring = scoring, cv = cv)
    results_performance.append(res)


    estimator_ = sklearn.clone(estimator)
    res = get_impute_most_freq(estimator_, scaler,  X_missing = X_missing, y_missing = y_missing, scoring = scoring, cv = cv)
    results_performance.append(res)

    estimator_ = sklearn.clone(estimator)
    res = get_impute_knn_score(estimator_, scaler, X_missing = X_missing, y_missing = y_missing, scoring = scoring, cv = cv)
    results_performance.append(res)

    estimator_ = sklearn.clone(estimator)
    res = get_impute_iterative(estimator_, scaler, X_missing = X_missing, y_missing = y_missing, scoring = scoring, cv = cv)
    results_performance.append(res)
    
    plot_different_imputation_techs(x_labels, results_performance, scoring, title = 'Imputation Techniques')

    
    return x_labels, results_performance
