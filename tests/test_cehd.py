'''
Unit tests for the chemical exposure health data cleaning process.

Notes:
------
The expected data are stored in Feather format to preserve the exact structure
and data types of the pandas DataFrame, including pandas-specific types such 
as `Categorical`. 
'''

import pandas as pd 
import pytest 

from config_management import UnifiedConfiguration
from raw_processing import cehd_cleaning

#region: config fixture
@pytest.fixture
def config():
    '''
    '''
    return UnifiedConfiguration()
#endregion

#region: raw_exposure_data fixture
@pytest.fixture
def raw_exposure_data(config):
    '''
    '''
    return pd.read_csv(
        config.path['cehd_file'], 
        index_col=0,
        dtype=config.cehd['dtype']
    )
#endregion

#region: test_cehd_cleaning
def test_cehd_cleaning(raw_exposure_data, config):
    '''
    Use `pd.testing.assert_frame_equal` to compare the cleaned DataFrame with 
    the expected DataFrame
    
    Notes
    -----
    Feather does not serialize DataFrame indexes by default. To work around 
    this limitation, the DataFrame index is reset before saving to Feather and
    then restored after reading.
    '''
    expected_cehd_file = 'tests/expected_cehd.feather'
    expected_data = pd.read_feather(expected_cehd_file).set_index('index')

    cehd_data = cehd_cleaning.clean_chemical_exposure_health_data(
        raw_exposure_data, 
        config.path
        )
    
    pd.testing.assert_frame_equal(cehd_data, expected_data, check_names=False)
#endregion