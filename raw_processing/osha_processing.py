'''
'''

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24

# TODO: Test the code. Write docstrings. Get one-hot features

#region: prepare_target_from_raw
def prepare_target_from_raw(data_cleaner, twa_func):
    '''
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
    # A sampling event may involve several sampling numbers/workers
    # In such cases, aggregate the TWAs
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
    Extract the first N digits of the NAICS code based on the specified level.    
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
    Calculate the continuous exposure concentration (EC) using the EPA formula.
    '''
    AT = float(ED * DAYS_PER_YEAR * HOURS_PER_DAY)
    return (CA * ET * EF * ED) / AT
#endregion