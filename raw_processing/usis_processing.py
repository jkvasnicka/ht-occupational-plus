'''
'''

from raw_processing import osha_processing

#region: prepare_target_from_raw
def prepare_target_from_raw(usis_cleaner):
    '''
    '''
    y_for_naics = osha_processing.prepare_target_from_raw(
        usis_cleaner, 
        full_shift_twa_per_sampling
    )
    return y_for_naics
#endregion

# TODO: Remove hardcoded column names?
# TODO: Add data validation checks
#region: full_shift_twa_per_sampling
def full_shift_twa_per_sampling(exposure_data):
    '''
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