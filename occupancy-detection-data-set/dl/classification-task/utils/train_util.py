from pprint import pprint

import itertools

import numpy as np
import pandas as pd

import os
import sys
from time import time

import matplotlib.pyplot as plt

import sklearn

from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn import set_config
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoLars
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)

from sklearn.compose import make_column_selector as selector
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline


from utils.build_models import build_model
from utils.cross_validation_train import k_fold_cv_classification
from utils.plot_util import show_cm


def train_and_test(train_dataset, test_dataset, n_features, n_classes, target_names, num_epochs = 5, batch_size = 1, verbose = 0):

    x_train, y_train = train_dataset
    x_test, y_test = test_dataset

    model = build_model(n_features = n_features, n_classes = n_classes)
    model.fit(
        x_train, y_train,
        epochs = num_epochs,
        batch_size = batch_size,
        verbose = verbose
    )
    
    results = model.evaluate(x_test, y_test)
    print(results)

    if n_classes > 0:
        y_pred = model.predict_classes(x_test)
        # matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        show_cm(y_test, y_pred, target_names, n_classes = n_classes, title = 'Confusion Matrix - Occupancy Dataset')
        pprint(classification_report(y_test, y_pred, target_names = target_names))
        pass
    
    pass


def bench_k_means(estimator, name, data, labels, sample_size = 300):
    
    t0 = time()
    estimator.fit(data)
    results = '%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' \
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size))
    return results


def wrapper_bench_k_means(data, n_clusters, labels, n_init = 10, sample_size = 300, random_state = 42):

    results_list = list()

    result = bench_k_means(KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = n_init, random_state = random_state),
              name = "k-means++", data = data, sample_size = sample_size, labels = labels)
    results_list.append(result)

    result = bench_k_means(KMeans(init = 'random', n_clusters = n_clusters, n_init = n_init, random_state = random_state),
              name = "random", data = data, sample_size = sample_size, labels = labels)
    results_list.append(result)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_clusters).fit(data)
    result = bench_k_means(KMeans(init = pca.components_, n_clusters = n_clusters, n_init = 1),
              name = "PCA-based",
              data = data, sample_size = sample_size, labels = labels)
    results_list.append(result)


    header_str = 'init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette'
    data = [res.split('\t') for res in results_list]
    columns = list(filter(lambda xx: xx != "", header_str.split('\t')))

    df_results = pd.DataFrame(data = data, columns = columns)

    print(85 * '_')
    print(df_results.head(df_results.shape[0]))
    print(85 * '_')
    pass


def create_and_run_pipeline(X, y, clf_obj = LogisticRegression(), numeric_features = None, categorical_features = None, strategy_numeric = 'median', strategy_categorical = 'constant'):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy_numeric)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy_categorical, fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', clf_obj)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))
    
    return clf


def create_and_run_pipeline_GDCV(X, y, param_grid, num_cv = 10, clf_obj = LogisticRegression(), random_state = 42):

    # Reproduce the identical fit/score process
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'classifier__C': [0.1, 1.0, 10, 100],
    }

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="category")),
        ('cat', categorical_transformer, selector(dtype_include="category"))
    ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', clf_obj)])


    grid_search = GridSearchCV(clf, param_grid, cv=num_cv)
    grid_search.fit(X_train, y_train)

    clf_name = str(clf_obj).split('(')[0]
    print(
        ("best %s from grid search: %.3f"
        % (clf_name, grid_search.score(X_test, y_test)))
        )
    return clf


def create_and_run_pipeline_GDCV_v2(train_set, test_set, param_grid, n_folds = 10, clf_obj = LogisticRegression(), random_state = 42):

    # Reproduce the identical fit/score process
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)

    X_train, y_train = train_set.iloc[:,1:-1], train_set.iloc[:,-1]
    X_test, y_test = test_set.iloc[:,1:-1], test_set.iloc[:,-1]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="category")),
        ('cat', categorical_transformer, selector(dtype_include="category"))
    ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', clf_obj)])


    grid_search = GridSearchCV(clf, param_grid, cv=n_folds)
    grid_search.fit(X_train, y_train)

    clf_name = str(clf_obj).split('(')[0]
    print(
        ("best %s from grid search: %.3f"
        % (clf_name, grid_search.score(X_test, y_test)))
        )
    return clf, grid_search


def wrapper_feature_transformer_ensembles_trees_clf(X, y, X_test, y_test, n_estimator = 10):

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="category")),
        ('cat', categorical_transformer, selector(dtype_include="category"))
    ])

    scaler = StandardScaler()
    scaler.fit(X)
    x_train_scaled = scaler.transform(X)
    x_test_scaled = scaler.transform(X_test)

    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(
        x_train_scaled, y, test_size=0.5, random_state=0)

    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)
    rt_lm = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict(X_test)
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression(max_iter=1000)
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

    y_pred_rf_lm = rf_lm.predict(rf_enc.transform(rf.apply(x_test_scaled)))
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

    # Supervised transformation based on gradient boosted trees
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression(max_iter=1000)
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_grd_lm = grd_lm.predict(
        # grd_enc.transform(grd.apply(x_test_scaled)[:, :, 0]))[:, 1]
        grd_enc.transform(grd.apply(x_test_scaled)[:, :, 0]))
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

    # The gradient boosted model by itself
    y_pred_grd = grd.predict(x_test_scaled)
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

    # The random forest model by itself
    y_pred_rf = rf.predict(X_test)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()

    pass


def wrapper_feature_transformer_ensembles_trees_clf_v2(X, y, X_test, y_test, n_estimator = 10, clf_obj = LogisticRegression(max_iter=1000)):

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="category")),
        ('cat', categorical_transformer, selector(dtype_include="category"))
    ])

    scaler = StandardScaler()
    scaler.fit(X)
    x_train_scaled = scaler.transform(X)
    x_test_scaled = scaler.transform(X_test)

    clf_name = str(clf_obj).split('(')[0]

    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(
        x_train_scaled, y, test_size=0.5, random_state=0)

    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)
    rt_clf = sklearn.base.clone(clf_obj)
    pipeline = make_pipeline(rt, rt_clf)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict(X_test)
    fpr_rt_clf, tpr_rt_clf, _ = roc_curve(y_test, y_pred_rt)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder()
    rf_clf = sklearn.base.clone(clf_obj)
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_clf.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

    y_pred_rf_clf = rf_clf.predict(rf_enc.transform(rf.apply(x_test_scaled)))
    fpr_rf_clf, tpr_rf_clf, _ = roc_curve(y_test, y_pred_rf_clf)

    # Supervised transformation based on gradient boosted trees
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_clf = sklearn.base.clone(clf_obj)
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_clf.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_grd_clf = grd_clf.predict(
        # grd_enc.transform(grd.apply(x_test_scaled)[:, :, 0]))[:, 1]
        grd_enc.transform(grd.apply(x_test_scaled)[:, :, 0]))
    fpr_grd_clf, tpr_grd_clf, _ = roc_curve(y_test, y_pred_grd_clf)

    # The gradient boosted model by itself
    y_pred_grd = grd.predict(x_test_scaled)
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

    # The random forest model by itself
    y_pred_rf = rf.predict(X_test)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_clf, tpr_rt_clf, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_clf, tpr_rf_clf, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_clf, tpr_grd_clf, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve - {clf_name}')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_clf, tpr_rt_clf, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_clf, tpr_rf_clf, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_clf, tpr_grd_clf, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve (zoomed in at top left) - {clf_name}')
    plt.legend(loc='best')
    plt.show()

    pass
