'''
'''

import pandas as pd 

from . import osha_loading

#region: raw_usis_data
def raw_usis_data(raw_usis_file, initial_dtypes):
    '''
    '''
    exposure_data = pd.read_feather(raw_usis_file)
    # Apply minimal data cleaning for consistency with CEHD
    return pre_clean(exposure_data, initial_dtypes)
#endregion

#region: pre_clean
def pre_clean(exposure_data, initial_dtypes):
    '''Apply minimal data cleaning for consistency with CEHD'''
    exposure_data = exposure_data.copy()
    exposure_data = osha_loading.pre_clean(exposure_data, initial_dtypes)
    exposure_data['year'] = _extract_sample_year(exposure_data['sample_date'])
    return exposure_data
#endregion

#region: _extract_sample_year
def _extract_sample_year(sample_dates):
    '''
    Extract the sample year from the sample date.

    Creates a new column with the sample year.
    '''
    return (
        pd.to_datetime(sample_dates)
        .dt.year
        .astype('int64')
    )
#endregion