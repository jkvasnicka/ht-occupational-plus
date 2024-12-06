'''
This module prepares model variables specific to the CEHD dataset.
'''

from raw_processing import osha_processing

#region: target_from_raw
def target_from_raw(cehd_cleaner, write_dir=None):
    '''
    Prepares the target variable of exposure concentration for each unique 
    combination of chemical and NAICS code.
    '''
    y_for_naics = osha_processing.target_from_raw(
        cehd_cleaner, 
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
        time_sampled_col,
        chem_id_col, 
        naics_code_col, 
        inspection_number_col,
        sampling_number_col,
        **kwargs
        ):
    '''
    Returns a time-weighted average (TWA) concentration per sampling number.

    Notes
    -----
    Multiple records tied to a single sampling number and chemical agent 
    in CEHD are treated as sequential partial-shift measurements and are
    aggregated to calculate total sampling time and a TWA concentration 
    result for the evaluation (Sarazin et al., 2018).

    Returns
    -------
    pandas.Series
        The index is a MultiIndex with the following levels:
            0. chem_id_col
            1. naics_code_col
            3. inspection_number_col
            4. sampling_number_col
    The values are the TWAs of the 'sample_result_col' per group.

    Reference
    ---------
    Sarazin et al. (2018) - DOI: 10.1093/annweh/wxy003
    '''
    grouping_columns = [
        chem_id_col, 
        naics_code_col,
        inspection_number_col,
        sampling_number_col
    ]

    twa_per_sampling_number = (
        exposure_data
        .groupby(grouping_columns)
        .apply(lambda df: time_weighted_average(
            df[sample_result_col], 
            df[time_sampled_col])
            )
    )

    return twa_per_sampling_number
#endregion

#region: time_weighted_average
def time_weighted_average(ci, ti):
    '''
    Computes a Time-Weighted Average (TWA) concentration from partial-shift 
    samples.

    Parameters
    ----------
    ci : array-like
        Sequential, partial-shift concentrations
    ti : array-like
        Corresponding sampling times

    Returns
    -------
    float
    '''
    return sum(ci*ti) / sum(ti)
#endregion