'''
'''

import os
import pandas as pd

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

#region: read_target
def read_target(target_file):
    ''''''
    return pd.read_csv(target_file, index_col=[0, 1]).squeeze()
#endregion

#region: read_features
def read_features(opera_features_file, y=None):
    '''
    '''
    X_opera = pd.read_parquet(opera_features_file)
    X_naics = naics_features_from_target(y)
    return X_naics.join(X_opera, on='DTXSID', how='inner')
#endregion

#region: naics_features_from_target
def naics_features_from_target(y):
    '''
    Prepares one-hot-encoded NAICS codes as features.

    The features are derived from the target variable's index.
    '''
    naics_code_col = y.index.names[-1]
    X = pd.get_dummies(y.reset_index()[naics_code_col])
    return X.set_index(y.index).rename(columns=lambda col : str(col))
#endregion