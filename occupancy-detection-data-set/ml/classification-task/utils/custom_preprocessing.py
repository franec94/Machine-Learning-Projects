from utils.libs import *


def get_scaler_continuous_features(scaler_strategy):

    strategie_names = ["standardize", "normalize"]
    strategy_objs = [StandardScaler(), Normalizer()]

    strategies_dict = dict(zip(strategie_names, strategy_objs))
    return strategies_dict[scaler_strategy]


def select_chi2(select_chi2, X_train, y_train, X_test, feature_names):
    if select_chi2:
        print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
            print("done in %fs" % (time() - t0))
            print()
    return X_train, X_test, feature_names


def preprocess_data(train_data, test_data, opts, features_idx, targets_idx, strategy = 'mean', random_state = 0):

    feature_names = []
    transformers = []

    # Get Feature Matrix and Target from train dataset
    X_train = train_data.iloc[:, features_idx]
    y_train = train_data.iloc[:, targets_idx]

    feature_names = list(train_data.iloc[:, features_idx].columns)

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

    # Get Feature Matrix and Target from test dataset
    X_test = test_data.iloc[:, features_idx]
    y_test = test_data.iloc[:, targets_idx]

    # Transformer continuous predictors
    scaler = get_scaler_continuous_features(opts)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = strategy)),
        ('scaler', scaler)
    ])
    transformers.append(('num', numeric_transformer, selector(dtype_exclude="category")))

    # Transformer categorical predictors 
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    transformers.append(('cat', categorical_transformer, selector(dtype_include="category")))

    # Preprocess Data
    preprocessor = ColumnTransformer(transformers=transformers)

    t0 = time()
    preprocessor.fit(X_train)   
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Check wether to select first nmost -informative features by means of chi2 test
    X_train, X_test, feature_names = select_chi2(opts, X_train, y_train, X_test, feature_names)

    print("Preprocessing done in %fs" % (time() - t0))
    return X_train, y_train, X_test, y_test, feature_names
