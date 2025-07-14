'''
This module provides utility functions specific to the CompTox Chemistry
Dashboard.
'''

import pandas as pd 

#region: data_from_raw
def data_from_raw(comptox_file):
    '''
    Load and clean data from the CompTox Chemistry Dashboard.

    Parameters
    ----------
    comptox_file : str
        Path to the CSV file containing raw CompTox data.

    Returns
    -------
    pd.DataFrame
    '''
    comptox_data = pd.read_csv(comptox_file)

    comptox_data = comptox_data.drop_duplicates()

    return comptox_data
#endregion