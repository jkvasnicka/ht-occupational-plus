'''
Unit tests for the United States Information System (USIS) data cleaning. 
'''

import pandas as pd 
import pytest

from config_management import UnifiedConfiguration
from raw_processing import usis_cleaning

#region: config fixture
@pytest.fixture
def config():
    '''Fixture to load the unified configuration.'''
    return UnifiedConfiguration()
#endregion

#region: test_usis_data fixture
@pytest.fixture
def test_usis_data(config):
    '''Fixture to load the expected data for comparison.'''
    return load_data_from_feather(config.path['test_usis_file'])
#endregion

#region: raw_usis_data fixture
@pytest.fixture
def raw_usis_data(config):
    '''Fixture to load the raw USIS data for cleaning'''
    return load_data_from_feather(config.path['raw_usis_file'])
#endregion

#region: load_data_from_feather
def load_data_from_feather(file):
    '''Load expected data for comparison.'''
    data = pd.read_feather(file)
    if 'index' in data.columns:
        # The index was reset prior to writing to Feather
        data = data.set_index('index')
    return data
#endregion

#region: test_usis_cleaning
def test_usis_cleaning(raw_usis_data, test_usis_data, config):
    '''
    Use `pd.testing.assert_frame_equal` to compare the cleaned DataFrame with 
    the expected DataFrame.
    '''
    cleaner = usis_cleaning.UsisCleaner(config.path, config.usis)

    usis_data = cleaner.clean_raw_data(raw_usis_data)

    pd.testing.assert_frame_equal(usis_data, test_usis_data, check_names=False)
#endregion