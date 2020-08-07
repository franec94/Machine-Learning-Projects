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

import os

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from pprint import pprint

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
from keras.wrappers.scikit_learn import KerasClassifier

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

# =============================================================================================== #
# Custom Packahges
# =============================================================================================== #

from utils.build_models import build_model

from utils.cross_validation_train import k_fold_cv_classification
from utils.plot_util import show_cm, visualize_pca_reduced_data

from utils.train_util import wrapper_bench_k_means, wrapper_feature_transformer_ensembles_trees_clf, wrapper_feature_transformer_ensembles_trees_clf_v2
from utils.train_util import create_and_run_pipeline, create_and_run_pipeline_GDCV, create_and_run_pipeline_GDCV_v2
