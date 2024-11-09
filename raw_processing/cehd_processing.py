'''
'''

from raw_processing import osha_processing

#region: prepare_target_from_raw
def prepare_target_from_raw(cehd_cleaner):
    '''
    '''
    y_for_naics = osha_processing.prepare_target_from_raw(
        cehd_cleaner, 
        full_shift_twa_per_sampling
    )
    return y_for_naics
#endregion

# TODO: Remove hardcoded column names?
# TODO: Add data validation checks
#region: full_shift_twa_per_sampling
def full_shift_twa_per_sampling(exposure_data):
    '''
    
    Notes
    -----
    Multiple records tied to a single sampling number and chemical agent 
    in CEHD are treated as sequential partial-shift measurements and 
    aggregated to calculate total sampling time and a TWA concentration 
    result for the evaluation (Sarazin et al., 2018).

    Reference
    ---------
    Sarazin et al. (2018) - DOI: 10.1093/annweh/wxy003
    '''
    grouping_columns = [
        'DTXSID', 
        'NAICS_CODE',
        'INSPECTION_NUMBER',
        'SAMPLING_NUMBER'
    ]

    twa_per_sampling_number = (
        exposure_data
        .groupby(grouping_columns)
        .apply(lambda df: time_weighted_average(
            df['SAMPLE_RESULT'], 
            df['TIME_SAMPLED'])
            )
        .droplevel('SAMPLING_NUMBER')
    )

    return twa_per_sampling_number
#endregion

#region: time_weighted_average
def time_weighted_average(ci, ti):
    '''
    Compute the Time-Weighted Average (TWA) concentration. 

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