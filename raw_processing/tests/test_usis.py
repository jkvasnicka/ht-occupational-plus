'''
Unit tests for the United States Information System (USIS) data cleaning. 
'''

import pandas as pd 
import pytest

from config_management import UnifiedConfiguration
from raw_processing import usis_cleaning

# TODO: This looks redundant to test_cehd

#region: config fixture
@pytest.fixture
def config():
    '''Fixture to load the unified configuration.'''
    return UnifiedConfiguration()
#endregion

#region: cleaner fixture
@pytest.fixture
def cleaner(config):
    '''Fixture to instantiate the data cleaner'''
    return usis_cleaning.UsisCleaner(config.usis, config.path)
#endregion

#region: raw_exposure_data fixture
@pytest.fixture
def raw_exposure_data(cleaner):
    '''Fixture to load the raw USIS data for cleaning'''
    return cleaner.load_raw_data()
#endregion

#region: test_data fixture
@pytest.fixture
def test_data(config):
    '''Fixture to load the expected data for comparison.'''
    return load_test_data(config.path)
#endregion

#region: load_test_data
def load_test_data(path_settings):
    '''Load expected data from a Feather file and restore the index.'''
    return pd.read_feather(path_settings['test_usis_file']).set_index('index')
#endregion

#region: test_usis_cleaning
def test_usis_cleaning(cleaner, raw_exposure_data, test_data, config):
    '''
    Use `pd.testing.assert_frame_equal` to compare the cleaned DataFrame with 
    the expected DataFrame.
    '''
    usis_data = cleaner.clean_raw_data(
        raw_exposure_data,
        log_file=config.path['usis_log_file']
        )

    pd.testing.assert_frame_equal(usis_data, test_data, check_names=False)
#endregion