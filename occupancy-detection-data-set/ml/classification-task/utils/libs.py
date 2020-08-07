# =============================================================================================== #
# Standard Packages
# =============================================================================================== #
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

import itertools
import re

import numpy as np
import pandas as pd

import logging
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from optparse import OptionParser
from pprint import pprint
from time import time

# =============================================================================================== #
# Base Machine Learning Packages
# =============================================================================================== #

import keras
import sklearn
import tensorflow as tf

# =============================================================================================== #
# Sklearn Packages
# =============================================================================================== #


from sklearn import random_projection

from sklearn import set_config

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoLars
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)

from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# =============================================================================================== #
# Custom Packahges
# =============================================================================================== #

from utils.build_models import build_model

from utils.cross_validation_train import k_fold_cv_classification
from utils.plot_util import show_cm, visualize_pca_reduced_data, show_roc_curve, get_axes

from utils.train_util import wrapper_bench_k_means, wrapper_feature_transformer_ensembles_trees_clf, wrapper_feature_transformer_ensembles_trees_clf_v2
from utils.train_util import create_and_run_pipeline, create_and_run_pipeline_GDCV, create_and_run_pipeline_GDCV_v2
from utils.benchmark_util import add_plot_banchmark, wrapper_bechmark
from utils.custom_options import get_option_parser
from utils.custom_preprocessing import preprocess_data
from utils.benchmark_imputing_missing_values import benchmark_imputing_missing_values