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
    '''
    '''
    return UnifiedConfiguration()
#endregion

#region: test_usis_cleaning
def test_usis_cleaning(config):
    '''
    '''
    expected_usis_file = 'raw_processing/tests/expected_usis.feather'
    expected_data = pd.read_feather(expected_usis_file).set_index('index')

    raw_usis_data = pd.read_feather(config.path['raw_usis_file'])
    usis_data = usis_cleaning.clean_usis_data(raw_usis_data)

    pd.testing.assert_frame_equal(usis_data, expected_data, check_names=False)
#endregion