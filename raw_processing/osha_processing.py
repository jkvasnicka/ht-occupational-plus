'''
'''

from os import path 

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24

#region: exposure_targets_by_naics
def exposure_targets_by_naics(
        twa_per_sampling_number, 
        naics_levels,
        naics_code_col,
        chem_id_col,
        inspection_number_col,
        write_dir=None):
    '''
    Orchestrates target variable preparation for OSHA datasets.

    Calculates a representative exposure concentration for each unique 
    combination of chemical and NAICS code. First, a time-weighted average 
    (TWA) is calculated across any partial-shift measurements. Then, the TWAs
    are aggregated across sampling numbers (unique workers) and industries 
    within a given NAICS. Lastly, the TWAs are converted to continuous 
    equivalents, representative of chronic exposure and directly comparable to
    a human-equivalent point of departure (POD).
    '''
    # Convert the Series to a DataFrame to modify the index values (NAICS)
    original_index = list(twa_per_sampling_number.index.names)
    twa_df = twa_per_sampling_number.reset_index()

    y_for_naics = {}  # initialize
    for level in naics_levels:

        ## Replace the full NAICS codes with the specified NAICS level
        # assign() returns a copy, and dict unpacking ensures that 
        # 'naics_code_col' is interpreted as a variable rather than a string 
        kwargs = {
            naics_code_col: extract_naics_level(
                twa_df[naics_code_col], 
                level=level)
                }
        new_twa_per_sampling_number = (
            twa_df.assign(**kwargs)
            .set_index(original_index)
            .squeeze()
        )

        y_for_naics[level] = exposure_concentration_per_naics(
            new_twa_per_sampling_number,
            chem_id_col,
            naics_code_col,
            inspection_number_col
        )

        if write_dir:
            write_target(y_for_naics[level], write_dir, level)

    return y_for_naics
#endregion

#region: exposure_concentration_per_naics
def exposure_concentration_per_naics(
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

    return ec_per_naics.rename('mg_per_m3')
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

    References
    ----------
    Environmental Protection Agency (EPA). (2009). Risk assessment guidance 
    for superfund: Volume I: Human health evaluation manual (Part F, 
    supplemental guidance for inhalation risk assessment) (EPA-540-R-070-002; 
    OSWER 9285.7-82). Office of Superfund Remediation and Technology 
    Innovation.

    Notes
    -----
    The calculation aligns with EPA guidelines for chronic or subchronic 
    risk assessment (Equation 8 of the document).
    '''
    AT = float(ED * DAYS_PER_YEAR * HOURS_PER_DAY)
    return (CA * ET * EF * ED) / AT
#endregion

# TODO: Incorporate utilities.ensure_directory_exists()
#region: write_target
def write_target(y, write_dir, naics_level):
    '''Write the target variable to a file.'''
    file_name = f'{naics_level}.csv'
    target_file = path.join(write_dir, file_name)
    y.reset_index().to_csv(target_file, index=False)
#endregion