'''
Unit tests for the chemical exposure health data cleaning process.

THIS MODULE IS "UNDER CONSTRUCTION" AND MAY NOT BE UP-TO-DATE.
'''

import pandas as pd 
import pytest 

from config_management import UnifiedConfiguration
from raw_processing.cehd_cleaning import CehdCleaner

# TODO: Refactor this to look like osha_cleaning.prepare_clean_exposure_data

#region: config fixture
@pytest.fixture
def config():
    '''Fixture to load the unified configuration.'''
    return UnifiedConfiguration()
#endregion

#region: raw_exposure_data fixture
@pytest.fixture
def raw_exposure_data(cleaner):
    '''Fixture to load the raw chemical exposure health data.'''
    return cleaner.load_raw_data()
#endregion

#region: cleaner fixture
@pytest.fixture
def cleaner(config):
    '''Fixture to instantiate the data cleaner'''
    return CehdCleaner(config.cehd, config.path, config.comptox)
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
    return pd.read_feather(path_settings['test_cehd_file']).set_index('index')
#endregion

#region: test_cehd_cleaning
def test_cehd_cleaning(cleaner, raw_exposure_data, test_data, config):
    '''
    Use `pd.testing.assert_frame_equal` to compare the cleaned DataFrame with 
    the expected DataFrame.
    
    Notes
    -----
    Feather does not serialize DataFrame indexes by default. To work around 
    this limitation, the DataFrame index must be reset before writing to 
    Feather and then restored after reading.
    '''
    cehd_data = cleaner.clean_raw_data(
        raw_exposure_data,
        log_file=config.path['cehd_log_file']
    )
    
    pd.testing.assert_frame_equal(cehd_data, test_data, check_names=False)
#endregion