'''
'''

from raw_processing.osha_processing import target_file_path
import pandas as pd

#region: read_target
def read_target(target_dir, naics_level):
    ''''''
    target_file = target_file_path(target_dir, naics_level)
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
    return X.set_index(y.index)
#endregion