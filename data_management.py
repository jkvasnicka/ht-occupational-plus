'''
'''

import os
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
    '''
    X = read_features(
        features_file, 
        feature_columns=feature_columns,
        log10_features=log10_features
        )
    
    y = read_target(target_file)

    return X.align(y, join='inner', axis=0)
#endregion

# TODO: Add parameter validation, e.g., 'log10_features' in 'feature_columns'
#region: read_features
def read_features(features_file, feature_columns=None, log10_features=None):
    '''
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
    '''
    return pd.read_csv(target_file, index_col=[0, 1]).squeeze()
#endregion

# NOTE: This may be obsolete if only one target ends up being used
#region: read_targets
def read_targets(root_dir):
    '''
    Traverse a directory tree and read target data from CSV files.

    Returns
    -------
    dict
        A dictionary where keys represent the relative path to each file:
        - Simple strings for files in the root directory.
        - Tuples of directory names and file name (without extension) for 
          nested files.
        Values are the processed target data from each CSV file.
    '''
    data_for_name = {}  # initialize
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith('.csv'):
                relative_path = os.path.relpath(dirpath, root_dir)
                if relative_path == '.':  # root
                    path_parts = ()
                else:
                    path_parts = tuple(relative_path.split(os.sep))
                k = os.path.splitext(name)[0]  # file name without extension
                if path_parts:
                    k = tuple(path_parts + (k,))
                target_file = os.path.join(dirpath, name)
                data_for_name[k] = read_target(target_file)
    return data_for_name
#endregion