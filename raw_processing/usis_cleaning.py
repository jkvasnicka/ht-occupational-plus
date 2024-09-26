'''
'''

import pandas as pd

# TODO: Move to config file
CLEANING_STEPS = [
    'remove_nonpersonal',
    # 'clean_duplicates'
]

# TODO: Incorporate change log
# TODO: Remove short-duration samples?
#region: clean_usis_data
def clean_usis_data(usis_data):
    '''
    '''
    usis_data = pre_clean(usis_data)

    for step_name in CLEANING_STEPS:
        # TODO: Make this a common utility function in osha.py?
        usis_data = _apply_cleaning_step(usis_data, step_name)

    return usis_data
#endregion

#region: _apply_cleaning_step
def _apply_cleaning_step(exposure_data, step_name):
    '''
    '''
    return globals()[step_name](exposure_data)
#endregion

# TODO: This could be common function with cehd_cleaning.py
#region: remove_nonpersonal
def remove_nonpersonal(usis_data):
    '''
    Exclude all samples that are non-personal (e.g., area, etc.)
    '''
    usis_data = usis_data.copy()
    where_nonpersonal = usis_data['sample_type_id'] != 'P'
    return usis_data.loc[~where_nonpersonal]
#endregion

#region: pre_clean
def pre_clean(usis_data):
    '''
    '''
    usis_data = usis_data.copy()
    usis_data['year'] = _extract_sample_year(usis_data['sample_date'])
    return usis_data
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