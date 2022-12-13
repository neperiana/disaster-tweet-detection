import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score


def parameter_search(model_class, param_grid, n_splits, X, y):
    """
    Applies corss-validation to search for best parameters for a certain model.

    Parameters
    ----------
    model_class : Estimator class
        Sklearn estimator class to be used.
    param_grid : string
        Dictionary of parameter options.
    n_splits: int
        Number of splits in cross-validation.
    X : pandas.Dataframe (n_samples, n_columns)
        Training/validation set.
    y : array-like of shape (n_samples,)
        Array of original class labels per sample.

    Returns
    -------
    summary : pandas.Dataframe
        Returns the pre-processed string of text.
    """
    print(f'Searching best params for {model_class.__name__}...')
    results = []

    # Get cross validation indices
    skf = StratifiedKFold(n_splits=n_splits)
    print(f'No folds = {n_splits}')

    for i, (train, val) in enumerate(skf.split(X, y)):
        print(f'\nFold {i+1}/{n_splits}')

        # Split train/val sets
        X_train = X.iloc[train, :]
        y_train = y[train]
        X_val = X.iloc[val, :]
        y_val = y[val]
        scale_pos_weight = sum(np.where(y_train==0, 1, 0)) / sum(y_train)

        # Get search loop
        param_names = list(param_grid.keys())
        param_loop = ParameterGrid(param_grid)
        no_params = len(list(param_loop))
        print(f'Searching across {no_params} candidates')

        for params in param_loop:
            # Records params
            this_result = params.copy()
            print('.', end='')

            # Extract text features
            max_features = params.pop('max_features', None)
            vectorizer = TfidfVectorizer(max_features=max_features) 
            X_train_dtm = vectorizer.fit_transform(X_train['clean_text'])
            X_val_dtm = vectorizer.transform(X_val['clean_text'])

            # Train model
            clf = model_class(scale_pos_weight=scale_pos_weight, **params)
            clf.fit(X_train_dtm.toarray(), y_train)

            # Get predictions
            y_val_pred = clf.predict(X_val_dtm.toarray())

            # Score predictions
            this_result['tn'], this_result['fp'], this_result['fn'], this_result['tp'] = confusion_matrix(
                y_val, 
                y_val_pred,
                normalize='true',
            ).ravel()
            this_result['f1'] = f1_score(y_val, y_val_pred)
            this_result['auc'] = roc_auc_score(y_val, y_val_pred)

            results += [this_result]

    # Summarise results
    results = pd.DataFrame(results)
    return results.groupby(list(param_names), as_index=False)[['tn','tp', 'f1', 'auc']].mean()
        