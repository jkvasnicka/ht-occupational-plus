'''
This module prepares a full-shift time-weighted average (TWA) concentration
per sampling number specific to the USIS dataset. 

These data are eventually used to prepare a target variable for modeling.

See Also
--------
osha_processing.py
    Target variable preparation combining CEHD and USIS.
'''

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