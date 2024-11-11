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

#region: naics_features_from_target
def naics_features_from_target(y, naics_code_col):
    '''
    Prepares one-hot-encoded NAICS codes as features.

    The features are derived from the target variable's index.
    '''
    X = pd.get_dummies(y.reset_index()[naics_code_col])
    return X.set_index(y.index)
#endregion