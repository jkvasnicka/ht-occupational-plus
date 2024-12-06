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