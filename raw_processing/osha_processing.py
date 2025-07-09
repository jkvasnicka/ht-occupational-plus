'''
This module provides functions to prepare a target variable combining exposure
concentration data from both OSHA datasets (USIS and CEHD).
'''

import pandas as pd
from os import path 

from . import usis_cleaning, cehd_cleaning
from . import usis_processing, cehd_processing

# TODO: Add validation tests to ensure no duplicates in the index

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24

#region: combined_targets_from_raw
def combined_targets_from_raw(
        usis_settings, 
        cehd_settings, 
        path_settings, 
        comptox_settings=None,
        write_dir=None
        ):
    '''
    Prepare combined target variables from raw CEHD and USIS datasets.

    Returns
    -------
    dict of pandas.Series
        One target array for each NAICS level specified.
    '''
    usis_cleaner = usis_cleaning.UsisCleaner(
        usis_settings, 
        path_settings, 
        comptox_settings
        )
    usis_data = usis_cleaner.clean_exposure_data()

    cehd_cleaner = cehd_cleaning.CehdCleaner(
        cehd_settings, 
        path_settings, 
        comptox_settings
        )
    cehd_data = cehd_cleaner.clean_exposure_data()

    y_for_naics = combined_targets_from_data(
        usis_data, 
        cehd_data, 
        usis_settings, 
        cehd_settings, 
        write_dir=write_dir
        )

    return y_for_naics
#endregion

#region: combined_targets_from_data
def combined_targets_from_data(
        usis_data, 
        cehd_data, 
        usis_settings, 
        cehd_settings, 
        write_dir=None
        ):
    '''
    Prepare combined target variables from pre-cleaned CEHD and USIS data.

    Returns
    -------
    dict of pandas.Series
        One target array for each NAICS level specified.
    '''
    twa_usis = usis_processing.full_shift_twa_per_sampling(
        usis_data, 
        **usis_settings
        )
    twa_cehd = cehd_processing.full_shift_twa_per_sampling(
        cehd_data, 
        **cehd_settings
        )

    twa_per_sampling_number = combine_exposure_datasets(twa_usis, twa_cehd)

    y_for_naics = exposure_targets_by_naics(
        twa_per_sampling_number, 
        usis_settings['naics_levels'],
        usis_settings['naics_code_col'],
        usis_settings['chem_id_col'],
        usis_settings['inspection_number_col'],
        write_dir=write_dir
    )

    return y_for_naics
#endregion

#region: combine_exposure_datasets
def combine_exposure_datasets(twa_usis, twa_cehd):
    '''
    Combine CEHD and USIS datasets, using USIS data as the primary dataset and 
    adding CEHD data where no USIS data exists.

    Parameters
    ----------
    twa_usis, twa_cehd : pd.Series
        Time-weighted average (TWA) concentration per sampling number for USIS
        and CEHD, respectively.
    '''
    combined_series = twa_usis.copy()

    twa_cehd_additional = twa_cehd.loc[twa_cehd.index.difference(twa_usis.index)]

    return pd.concat([combined_series, twa_cehd_additional])
#endregion

#region: targets_from_raw
def targets_from_raw(
        data_cleaner,
        twa_function, 
        write_dir=None
        ):
    '''
    Prepare target variables from raw dataset input (USIS or CEHD).
    '''
    exposure_data = data_cleaner.clean_exposure_data()
    data_settings = data_cleaner.data_settings 

    y_for_naics = targets_from_data(
        exposure_data, 
        twa_function,
        data_settings,
        write_dir=write_dir
        )
    
    return y_for_naics
#endregion

#region: targets_from_data
def targets_from_data(
        exposure_data, 
        twa_function,
        data_settings,
        write_dir=None
        ):
    '''
    Prepare target variables from pre-cleaned dataset input (USIS or CEHD).
    '''
    twa_per_sampling_number = twa_function(
        exposure_data, 
        **data_settings
        )
    
    y_for_naics = exposure_targets_by_naics(
        twa_per_sampling_number, 
        data_settings['naics_levels'],
        data_settings['naics_code_col'],
        data_settings['chem_id_col'],
        data_settings['inspection_number_col'],
        write_dir=write_dir
    )

    return y_for_naics
#endregion

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
    y_for_naics = {}  # initialize

    for level in naics_levels:

        new_twa_per_sampling_number = reindex_with_naics_level(
            twa_per_sampling_number, 
            naics_code_col, 
            level
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

#region: reindex_with_naics_level
def reindex_with_naics_level(twa_per_sampling_number, naics_code_col, level):
    '''
    Replace the full NAICS codes with the specified NAICS level.
    '''
    # Convert the Series to a DataFrame to modify the index values (NAICS)
    original_index = list(twa_per_sampling_number.index.names)
    twa_df = twa_per_sampling_number.reset_index()
    
    # NOTE: assign() returns a copy, and dict unpacking ensures that 
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

    return new_twa_per_sampling_number
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