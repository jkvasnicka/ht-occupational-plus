'''
'''

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24

#region: target_from_raw
def target_from_raw(data_cleaner, twa_func):
    '''
    Orchestrates target variable preparation for OSHA datasets.

    Calculates a representative exposure concentration for each unique 
    combination of chemical and NAICS code. First, a time-weighted average 
    (TWA) is calculated across any partial-shift measurements. Then, the TWAs
    are aggregated across sampling numbers (unique workers) and industries 
    within a given NAICS. Lastly, the TWAs are converted to continuous 
    equivalents, representative of chronic exposure and directly comparable to
    a margin of exposure.
    '''
    exposure_data = data_cleaner.prepare_clean_exposure_data()
    data_settings = data_cleaner.data_settings

    y_for_naics = {}  # initialize
    for level in data_settings['naics_levels']:

        # NOTE: assign() returns a copy with the specified NAICS level
        # Dict unpacking is used to interpret 'naics_code_col' as a variable
        # rather than a literal string.
        naics_code_col = data_settings['naics_code_col']
        kwargs = {
            naics_code_col: extract_naics_level(
                exposure_data[naics_code_col], 
                level=level)
                }
        twa_per_sampling_number = twa_func(exposure_data.assign(**kwargs))

        y_for_naics[level] = prepare_target(
            twa_per_sampling_number,
            data_settings['chem_id_col'],
            naics_code_col,
            data_settings['inspection_number_col']
        )

    return y_for_naics
#endregion

#region: prepare_target
def prepare_target(
        twa_per_sampling_number,
        chem_id_col,
        naics_code_col,
        inspection_number_col
        ):
    '''
    Converts TWAs per sampling number into final exposure concentration (EC)
    values.
    '''
    twa_per_inspection = aggregate_twa_per_inspection(
        twa_per_sampling_number,
        chem_id_col,
        naics_code_col,
        inspection_number_col
    )

    twa_per_naics = aggregate_twa_per_naics(
        twa_per_inspection,
        chem_id_col,
        naics_code_col
    )

    ec_per_naics = continuous_exposure_concentration(twa_per_naics)

    return ec_per_naics
#endregion

#region: aggregate_twa_per_naics
def aggregate_twa_per_naics(
        twa_per_inspection, 
        chem_id_col, 
        naics_code_col
        ):
    '''
    Aggregates TWAs across all OSHA inspections within a given NAICS code.
    
    The resulting concentration values represent the typical (median) 
    exposure for a given NAICS code.
    '''
    twa_per_naics = (
        twa_per_inspection
        .groupby([chem_id_col, naics_code_col])
        .median()
    )
    return twa_per_naics
#endregion

#region: aggregate_twa_per_inspection
def aggregate_twa_per_inspection(
        twa_per_sampling_number,
        chem_id_col,
        naics_code_col,
        inspection_number_col
        ):
    '''
    Aggregates TWAs across sampling numbers (workers) to produce a 
    representative TWA for each chemical-inspection pair.

    The resulting concentration value represents the typical (median) exposure
    of all workers within a given inspection.
    '''
    chem_naics_inspection = [
        chem_id_col, 
        naics_code_col,
        inspection_number_col
    ]
    twa_per_inspection = (
        twa_per_sampling_number
        .groupby(chem_naics_inspection)
        .median()
    )
    return twa_per_inspection
#endregion

# TODO: Add data validation checks, e.g., no leading/trailing whitespace, etc.
#region: extract_naics_level
def extract_naics_level(naics_series, level):
    '''
    Extracts the first N digits of the NAICS code based on the specified 
    level.
    '''
    digits_for_level = {
        'sector': 2,
        'subsector': 3,
        'industry_group': 4,
        'industry': 5,
        'national_industry': 6
    }

    return naics_series.str[:digits_for_level[level]]
#endregion

#region: continuous_exposure_concentration
def continuous_exposure_concentration(CA, ET=8, EF=250, ED=25):
    '''
    Converts a chemical's concentration in air into a continuous exposure
    concentration. 

    Notes
    -----
    The calculation aligns with EPA guidelines for chronic or subchronic 
    risk assessment.
    '''
    AT = float(ED * DAYS_PER_YEAR * HOURS_PER_DAY)
    return (CA * ET * EF * ED) / AT
#endregion