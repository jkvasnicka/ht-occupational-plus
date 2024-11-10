'''
'''

import pandas as pd 

#region: naics_features_from_target
def naics_features_from_target(y, naics_code_col):
    '''
    Prepares one-hot-encoded NAICS codes as features.

    The features are derived from the target variable's index.
    '''
    X = pd.get_dummies(y.reset_index()[naics_code_col])
    return X.set_index(y.index)
#endregion