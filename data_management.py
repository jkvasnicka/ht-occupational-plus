'''
Data I/O utilities for features and target variables.

Provides functions to load and preprocess feature dataframes and target
series from disk.
'''

import pandas as pd
import numpy as np 

#region: read_features_and_target
def read_features_and_target(
        features_file, 
        target_file,
        feature_columns=None, 
        log10_features=None
        ):
    '''
    Load feature matrix and target series, align on index.

    Parameters
    ----------
    features_file : str
        Path to features Parquet file.
    target_file : str
        Path to target CSV file (MultiIndex).
    feature_columns : list of str, optional
        Subset of columns to select from the features DataFrame.
    log10_features : list of str, optional
        Columns to log₁₀-transform after selection.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix aligned to y.
    y : pandas.Series
        Target series aligned to X.
    '''
    X = read_features(
        features_file, 
        feature_columns=feature_columns,
        log10_features=log10_features
        )
    
    y = read_target(target_file)

    return X.align(y, join='inner', axis=0)
#endregion

#region: read_features
def read_features(features_file, feature_columns=None, log10_features=None):
    '''
    Read and preprocess feature DataFrame.
    '''
    if log10_features is None:
        log10_features = []

    X = pd.read_parquet(features_file)

    if feature_columns:
        X = X[feature_columns]

    for feature in log10_features:
        X[feature] = np.log10(X[feature])
        
    return X
#endregion

#region: read_target
def read_target(target_file):
    '''
    Read target series from CSV with MultiIndex (DTXSID, naics_id).
    '''
    return pd.read_csv(target_file, index_col=[0, 1]).squeeze()
#endregion