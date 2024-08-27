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

from raw_processing import cehd_cleaning

# TODO: Move to config files
path_settings = {
    'cehd_file' : 'Input/Processed/osha_data.csv',
    'qualif_conv_file' : 'Input/Raw/OSHA/CEHD/CEHD1984_2018/cleaning scripts/Conversion tables/qualif_new_2020.csv',
    'unit_conv_file' : 'Input/Raw/OSHA/CEHD/CEHD1984_2018/cleaning scripts/Conversion tables/unit_conv_2020.csv',
    'it_directory' : 'Input/Raw/OSHA/CEHD/CEHD1984_2018/cleaning scripts/Conversion tables IT',
    'cehd_log_file' : 'cehd_log_file.json'
} 

cehd_settings = {
    'dtype' : {
        'CITY': 'str',
        'ESTABLISHMENT_NAME': 'str',
        'FIELD_NUMBER': 'str',
        'IMIS_SUBSTANCE_CODE' : 'str',
        'INSPECTION_NUMBER': 'str',
        'INSTRUMENT_TYPE': 'str',
        'NAICS_CODE': 'str',
        'QUALIFIER': 'str',
        'SAMPLING_NUMBER': 'str',
        'SUBSTANCE': 'str',
        'UNIT_OF_MEASUREMENT': 'str',
        'ZIP_CODE': 'str'
    }
}

@pytest.fixture
def raw_exposure_data():
    '''
    '''
    return pd.read_csv(
        path_settings['cehd_file'], 
        index_col=0,
        dtype=cehd_settings['dtype']
    )

#region: test_cehd_cleaning
def test_cehd_cleaning(raw_exposure_data):
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
        path_settings
        )
    
    pd.testing.assert_frame_equal(cehd_data, expected_data, check_names=False)
#endregion