'''
This module prepares model variables specific to the USIS dataset. 
'''

from raw_processing import osha_processing

#region: target_from_raw
def target_from_raw(usis_cleaner, write_dir=None):
    '''
    Prepares the target variable of exposure concentration for each unique 
    combination of chemical and NAICS code.
    '''
    y_for_naics = osha_processing.target_from_raw(
        usis_cleaner, 
        full_shift_twa_per_sampling,
        write_dir=write_dir
    )
    return y_for_naics
#endregion

# TODO: Remove hardcoded column names?
# TODO: Add data validation checks
#region: full_shift_twa_per_sampling
def full_shift_twa_per_sampling(exposure_data):
    '''
    Returns a time-weighted average (TWA) concentration per sampling number.

    Requires that the exposure data have been cleaned such that any samples
    not representing a full-shift TWA have been removed.
    '''
    chem_naics_inspection = [
        'DTXSID', 
        'naics_id',
        'inspection_number'
    ]

    twa_per_sampling_number = (
        exposure_data[chem_naics_inspection + ['exposure_level']]
        .set_index(chem_naics_inspection)
        .squeeze()
        .rename(None)
    )

    return twa_per_sampling_number
#endregion