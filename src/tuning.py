import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score

from src.preprocessing import TextToFeatures


def parameter_search(model_class, param_grid, X, y, n_splits=3, vect_type='tfidf-vec'):
    """
    Applies corss-validation to search for best parameters for a certain model.

    Parameters
    ----------
    model_class : Estimator class
        Sklearn estimator class to be used.
    param_grid : string
        Dictionary of parameter options.
    X : pandas.Dataframe (n_samples, n_columns)
        Training/validation set.
    y : array-like of shape (n_samples,)
        Array of original class labels per sample.
    n_splits: int
        Number of splits in cross-validation.
    vect_type: str, default='tfidf-vec'
        Type of feature extraction applied to text. 
        Either tf-idf or sentence-BERT.

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

            # Extract text features - BERT
            BERT_vectorizer = TextToFeatures(type='sentence-BERT')
            X_train_dtm_a = BERT_vectorizer.fit_transform(X_train['clean_text']).to_numpy()
            X_val_dtm_a = BERT_vectorizer.transform(X_val['clean_text']).to_numpy()

            # Extract text features - keywords
            max_features = params.pop('max_features', None)
            tfidf_vectorizer = TextToFeatures(type='count-vec')
            X_train_dtm_b = tfidf_vectorizer.fit_transform(X_train['keyword'])
            X_val_dtm_b = tfidf_vectorizer.transform(X_val['keyword'])

            # Join 
            X_train_dtm = np.concatenate([X_train_dtm_a, X_train_dtm_b], axis=1)
            X_val_dtm = np.concatenate([X_val_dtm_a, X_val_dtm_b], axis=1)

            # Train model
            clf = model_class(scale_pos_weight=scale_pos_weight, **params)
            clf.fit(X_train_dtm, y_train)

            # Get predictions
            y_val_pred = clf.predict(X_val_dtm)

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
        