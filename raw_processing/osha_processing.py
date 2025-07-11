'''
This module provides functions to prepare a target variable combining exposure
concentration data from both OSHA datasets (USIS and CEHD).
'''

import pandas as pd
import os 

from . import usis_cleaning, cehd_cleaning
from . import usis_processing, cehd_processing

# TODO: Add validation tests to ensure no duplicates in the index

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24

#region: target_from_raw
def target_from_raw(
        usis_settings, 
        cehd_settings, 
        path_settings, 
        comptox_settings,
        naics_level,
        write_dir=None
        ):
    '''
    Prepare target variable from raw CEHD and USIS datasets.

    Optionally writes the target to disk.

    Parameters
    ----------
    usis_settings : dict
        Config settings for the USIS dataset.
    cehd_settings : dict
        Config settings for the CEHD dataset.
    path_settings : dict
        Config settings for file paths.
    comptox_settings : dict
        Config settings for CompTox data.
    naics_level : str
        Level of the NAICS code at which to aggregate. Must be either 
        'sector', 'subsector', 'industry_group', 'industry', or
        'national_industry'.
    write_dir : str, optional
        Directory in which the results will be written.

    Returns
    -------
    pandas.Series
        Exposure target with MultiIndex (chem_id, naics_id).
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

    y = _target_from_data(
            usis_data, 
            cehd_data, 
            usis_settings, 
            cehd_settings,
            comptox_settings['chem_id_col'],
            naics_level
            )
    
    if write_dir:
        _write_target(y, write_dir, naics_level)

    return y
#endregion

#region: _target_from_data
def _target_from_data(
        usis_data, 
        cehd_data, 
        usis_settings, 
        cehd_settings,
        chem_id_col,
        naics_level
        ):
    '''
    Prepare target variable from pre-cleaned CEHD and USIS data.

    Parameters
    ----------
    usis_data : pandas.DataFrame
        Pre-cleaned USIS dataset.
    cehd_data : pandas.DataFrame
        Pre-cleaned CEHD dataset.
    usis_settings : dict
        Config settings for the USIS dataset.
    cehd_settings : dict
        Config settings for the CEHD dataset.
    chem_id_col : str
        Name of the column corresponding to the chemical identifiers.
    naics_level : str
        Level of the NAICS code at which to aggregate. Must be either 
        'sector', 'subsector', 'industry_group', 'industry', or
        'national_industry'.

    Returns
    -------
    pandas.Series
        Exposure target with MultiIndex (chem_id, naics_id).
    '''
    twa_usis = usis_processing.full_shift_twa_per_sampling(
        usis_data,
        chem_id_col,
        **usis_settings
        )
    twa_cehd = cehd_processing.full_shift_twa_per_sampling(
        cehd_data,
        chem_id_col,
        **cehd_settings
        )

    twa_combined = _combine_datasets(twa_usis, twa_cehd)

    y = _exposure_conc_from_twa(
            twa_combined,
            chem_id_col,
            naics_level,
            usis_settings['naics_code_col'],  # retains USIS name convention
            usis_settings['inspection_number_col']
        )

    return y
#endregion

#region: _combine_datasets
def _combine_datasets(twa_usis, twa_cehd):
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

#region: _exposure_conc_from_twa
def _exposure_conc_from_twa(
        twa_per_sampling_number,
        chem_id_col,
        naics_level,
        naics_code_col,
        inspection_number_col
        ):
    '''
    Converts time-weighted averaged (TWA) per sampling number to an exposure
    concentration.

    Calculates a representative exposure concentration for each unique 
    combination of chemical and NAICS code. First, a time-weighted average 
    (TWA) is calculated across any partial-shift measurements. Then, the TWAs
    are aggregated across sampling numbers (unique workers) and industries 
    within a given NAICS. Lastly, the TWAs are converted to continuous 
    equivalents, representative of chronic exposure and directly comparable to
    a human-equivalent point of departure (POD).
    '''
    new_twa_per_sampling_number = _replace_naics_with_level(
        twa_per_sampling_number, 
        naics_code_col, 
        naics_level
        )
    
    twa_per_inspection = _aggregate_twa_per_inspection(
        new_twa_per_sampling_number,
        chem_id_col,
        naics_code_col,
        inspection_number_col
    )

    twa_per_naics = _aggregate_twa_per_naics(
        twa_per_inspection,
        chem_id_col,
        naics_code_col
    )

    ec_per_naics = continuous_exposure(twa_per_naics)

    return ec_per_naics.rename('mg_per_m3')
#endregion

#region: _replace_naics_with_level
def _replace_naics_with_level(twa_per_sampling_number, naics_code_col, level):
    '''
    Replaces the index to show the specified NAICS level (e.g., sector) rather 
    than the full six-digit NAICS code.
    '''
    # Convert the Series to a DataFrame to modify the index values (NAICS)
    original_index = list(twa_per_sampling_number.index.names)
    twa_df = twa_per_sampling_number.reset_index()
    
    # NOTE: assign() returns a copy, and dict unpacking ensures that 
    # 'naics_code_col' is interpreted as a variable rather than a string 
    kwargs = {
        naics_code_col: _extract_naics_level(
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

#region: _aggregate_twa_per_naics
def _aggregate_twa_per_naics(
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

#region: _aggregate_twa_per_inspection
def _aggregate_twa_per_inspection(
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
#region: _extract_naics_level
def _extract_naics_level(naics_series, level):
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

#region: continuous_exposure
def continuous_exposure(CA, ET=8, EF=250, ED=25):
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

#region: _write_target
def _write_target(y, write_dir, naics_level):
    '''Write the target variable to disk as a CSV file.'''
    if not os.path.exists(write_dir):
        # Ensure directory exists
        os.makedirs(write_dir)
    file_name = f'{naics_level}.csv'
    target_file = os.path.join(write_dir, file_name)
    y.reset_index().to_csv(target_file, index=False)
#endregion