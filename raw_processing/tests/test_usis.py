'''
Unit tests for the United States Information System (USIS) data cleaning. 
'''

import pandas as pd 
import pytest

from config_management import UnifiedConfiguration
from raw_processing import usis

# TODO: This looks redundant to test_cehd

#region: config fixture
@pytest.fixture
def config():
    '''Fixture to load the unified configuration.'''
    return UnifiedConfiguration()
#endregion

#region: raw_exposure_data fixture
@pytest.fixture
def raw_exposure_data(config):
    '''Fixture to load the raw USIS data for cleaning'''
    return usis.load_raw_usis_data(config.path['raw_usis_file'])
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
def test_usis_cleaning(raw_exposure_data, test_data, config):
    '''
    Use `pd.testing.assert_frame_equal` to compare the cleaned DataFrame with 
    the expected DataFrame.
    '''
    cleaner = usis.UsisCleaner(config.usis, config.path)

    usis_data = cleaner.clean_raw_data(raw_exposure_data)

    pd.testing.assert_frame_equal(usis_data, test_data, check_names=False)
#endregion