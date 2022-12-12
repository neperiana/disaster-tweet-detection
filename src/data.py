import numpy as np
import pandas as pd
from src.preprocessing import preprocess_text


def get_data():
    """
    Get training dataset and preprocess text.

    Parameters
    ----------

    Returns
    -------
    X : pandas.Dataframe (n_train_samples, n_columns)
        Returns training set, including pre-processed text.
    y : array-like of shape (n_train_samples,)
        Array of original class labels per sample.
    test : pandas.Dataframe (n_test_samples, n_columns)
        Returns test set, including pre-processed text.
    """
    # Read dataset
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    # Clean text
    train['clean_text'] = train['text'].apply(preprocess_text)
    test['clean_text'] = test['text'].apply(preprocess_text)

    return train.drop('target', axis=1), train['target'], test
