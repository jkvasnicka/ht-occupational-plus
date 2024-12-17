'''
This module prepares model variables specific to the USIS dataset. 
'''

from raw_processing.usis_cleaning import UsisCleaner
from raw_processing import osha_processing

#region: exposure_targets_from_raw
def exposure_targets_from_raw(
        data_settings, 
        path_settings, 
        comptox_settings=None
        ):
    '''
    '''
    data_cleaner = UsisCleaner(data_settings, path_settings, comptox_settings)
    exposure_data = data_cleaner.prepare_clean_exposure_data()
    y_for_naics = osha_processing.prepare_exposure_targets(
        exposure_data, 
        full_shift_twa_per_sampling, 
        data_settings,
        write_dir=path_settings['usis_target_dir']
        )
    return y_for_naics
#endregion

# TODO: Add data validation checks
#region: full_shift_twa_per_sampling
def full_shift_twa_per_sampling(
        exposure_data,
        *,
        sample_result_col,
        chem_id_col, 
        naics_code_col, 
        inspection_number_col,
        sampling_number_col,
        **kwargs
        ):
    '''
    Returns a time-weighted average (TWA) concentration per sampling number.

    Requires that the exposure data have been cleaned such that any samples
    not representing a full-shift TWA have been removed.
    
    Returns
    -------
    pandas.Series
        The index is a MultiIndex with the following levels:
            0. chem_id_col
            1. naics_code_col
            3. inspection_number_col
            4. sampling_number_col
        The values are the TWAs of the 'sample_result_col' per group.
    '''
    grouping_columns = [
        chem_id_col, 
        naics_code_col,
        inspection_number_col,
        sampling_number_col
    ]

    twa_per_sampling_number = (
        exposure_data[grouping_columns + [sample_result_col]]
        .set_index(grouping_columns)
        .squeeze()
        .rename(None)
    )

    return twa_per_sampling_number
#endregion