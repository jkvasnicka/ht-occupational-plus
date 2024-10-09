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
from raw_processing import cehd_loading, cehd_cleaning

#region: config fixture
@pytest.fixture
def config():
    '''Fixture to load the unified configuration.'''
    return UnifiedConfiguration()
#endregion

#region: raw_exposure_data fixture
@pytest.fixture
def raw_exposure_data(config):
    '''Fixture to load the raw chemical exposure health data.'''
    return cehd_loading.raw_chem_exposure_health_data(config.cehd, config.path)
#endregion

#region: test_cehd_data fixture
@pytest.fixture
def test_cehd_data(config):
    '''Fixture to load the expected data for comparison.'''
    return load_test_data(config.path)
#endregion

#region: load_test_data
def load_test_data(path_settings):
    '''Load expected data from a Feather file and restore the index.'''
    return pd.read_feather(path_settings['test_cehd_file']).set_index('index')
#endregion

#region: test_cehd_cleaning
def test_cehd_cleaning(raw_exposure_data, test_cehd_data, config):
    '''
    Use `pd.testing.assert_frame_equal` to compare the cleaned DataFrame with 
    the expected DataFrame.
    
    Notes
    -----
    Feather does not serialize DataFrame indexes by default. To work around 
    this limitation, the DataFrame index must be reset before writing to 
    Feather and then restored after reading.
    '''
    cehd_data = cehd_cleaning.clean_chem_exposure_health_data(
        raw_exposure_data, 
        config.path,
        config.cehd
    )
    
    pd.testing.assert_frame_equal(cehd_data, test_cehd_data, check_names=False)
#endregion