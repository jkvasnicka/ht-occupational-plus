'''
This module implements a data cleaning methodology by Jérôme Lavoué - 
Université de Montréal for the Chemical Exposure Health Data (CEHD).

The original R script was translated to Python.
'''
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt 

CLEANING_STEPS = [
    'pre_clean',
    'remove_blanks',
    'remove_nonpersonal',
    'remove_rare_or_non_chemical',
    'replace_missing_values',
    'add_censored_column',
    'remove_invalid_non_detect',
    'clean_unit_of_measurement',
    'remove_blk_not_bulk',
    'remove_uninterpretable_qualifier',
    'remove_conflicting_qualifier',
    'remove_blk_possible_bulk_not_blank',
    'remove_combustion_related',
    'remove_fibers_substance_conflict',
    'remove_yttrium_substance_conflict',
    'remove_approximate_measure',
    'remove_qualifier_unit_mismatch',
    'remove_invalid_fibers_unit',
    'remove_empty_unit_non_null_result',
    'remove_percent_greater_than_100',
    'create_detection_indicator',
    'remove_invalid_unit',
    'convert_percent_to_mass_concentration',
    'remove_missing_office_identifier',
    'remove_missing_time_sampled',
    'remove_null_time_sampled',
    'remove_negative_sample_result',
    'remove_missing_sample_number',
    'remove_missing_volume',
    'remove_zero_volume_sampled',
    'clean_instrument_type',
    'clean_duplicates'
]

#region: clean_chemical_exposure_health_data
def clean_chemical_exposure_health_data(
        exposure_data, 
        path_settings, 
        do_log_changes=True
        ):
    '''
    '''
    change_log = {}  # initialize
    kwargs = _prepare_key_word_arguments(path_settings)

    for step_name in CLEANING_STEPS:
        N_before = len(exposure_data)
        exposure_data = _apply_cleaning_step(exposure_data, step_name, kwargs)
        N_after = len(exposure_data)
        change_log[step_name] =  N_after - N_before

    if do_log_changes is True:
        cehd_log_file = path_settings.get(
            'cehd_log_file', 'cehd_log_file.json'
            )
        with open(cehd_log_file, 'w') as log_file:
            json.dump(change_log, log_file, indent=4)

    return exposure_data
#endregion

#region: _prepare_key_word_arguments
def _prepare_key_word_arguments(path_settings):
    '''
    Define key-word arguments for flexible argument passing.
    '''
    kwargs = path_settings.copy()
    kwargs['qualif_conv_2020'] = load_qualifier_conversion(
        path_settings['qualif_conv_file']
        )
    kwargs['unit_conv_2020'] = load_unit_measure_conversion(
        path_settings['unit_conv_file']
    )
    return kwargs
#endregion

#region: _apply_cleaning_step
def _apply_cleaning_step(exposure_data, step_name, kwargs):
    '''
    '''
    return globals()[step_name](exposure_data, **kwargs)
#endregion

# TODO: Is this convoluted process necessary for removing duplicates?
#region: clean_duplicates
def clean_duplicates(exposure_data, **kwargs):
    '''
    Clean the dataset by identifying and removing duplicate samples.
    '''
    exposure_data = _create_hash(exposure_data)
    bla = _identify_potential_duplicates(exposure_data)
    false_duplicate_hashes = _identify_false_duplicates(exposure_data, bla)
    return _remove_true_duplicates(exposure_data, false_duplicate_hashes, bla)
#endregion

#region: _create_hash
def _create_hash(exposure_data):
    '''
    Create a unique HASH variable to identify potential duplicates.
    '''
    exposure_data = exposure_data.copy()
    exposure_data['HASH'] = (
        exposure_data['INSPECTION_NUMBER'].astype(str) + '-' +
        exposure_data['IMIS_SUBSTANCE_CODE'].astype(str) + '-' +
        exposure_data['SAMPLING_NUMBER'].astype(str) + '-' +
        exposure_data['FIELD_NUMBER'].astype(str)
    )
    return exposure_data
#endregion

#region: _identify_potential_duplicates
def _identify_potential_duplicates(exposure_data):
    '''
    Identify and return a DataFrame of potential duplicate records based on 
    the HASH variable.
    '''
    exposure_data = exposure_data.copy()

    bla = exposure_data['HASH'].value_counts().reset_index()
    bla.columns = ['name', 'n']
    bla = bla[bla['n'] > 1]
    bla['name'] = bla['name'].astype(str)

    # Match the values for 'code' and 'sub'
    bla['code'] = bla['name'].map(
        dict(zip(exposure_data['HASH'], exposure_data['IMIS_SUBSTANCE_CODE']))
    )
    bla['sub'] = bla['name'].map(
        dict(zip(exposure_data['HASH'], exposure_data['SUBSTANCE']))
    )

    # Ensure the order matches
    return bla.sort_values(by='name').reset_index(drop=True)
#endregion

#region: _identify_false_duplicates
def _identify_false_duplicates(exposure_data, bla):
    '''
    Identify false duplicates by comparing additional variables (CONCAT).
    
    False duplicates are identified where the CONCAT variable varies 
    for the same HASH.
    '''
    # Create a new hash variable to identify false duplicates
    exposure_data['CONCAT'] = (
        exposure_data['LAB_NUMBER'].astype(str) + '-' +
        exposure_data['STATE'].astype(str) + '-' +
        exposure_data['ZIP_CODE'].astype(str) + '-' +
        exposure_data['YEAR'].astype(str) + '-' +
        exposure_data['TIME_SAMPLED'].astype(str) + '-' +
        exposure_data['SAMPLE_WEIGHT_2'].astype(str)
    )

    # Identify samples where CONCAT is the same
    concat_counts = exposure_data.groupby('HASH')['CONCAT'].nunique()
    concatdiff_hashes = concat_counts.loc[concat_counts > 1].index
    bla['concatdiff'] = bla['name'].isin(concatdiff_hashes)

    # False duplicates occur where CONCAT varies
    return bla['name'].loc[bla['concatdiff']].to_numpy()
#endregion

# NOTE: Inconsistency
#region: _remove_true_duplicates
def _remove_true_duplicates(exposure_data, false_duplicate_hashes, bla):
    '''
    Remove true duplicates from the dataset, retaining only one sample per 
    duplicate.

    Notes
    -----
    The original R code hardcoded 'max_rows' to 6083, which resulted in the 
    R code inadvertently creating additional rows with NaN. This Python code
    addresses this issue by correctly defining 'max_rows' based on the length.
    '''
    exposure_data = exposure_data.copy()

    restrictM = exposure_data['HASH'].isin(false_duplicate_hashes)

    exposure_data_1 = exposure_data.loc[~restrictM]

    #### N: true duplicates ####
    # Separate the DB into the OK and remaining problematic
    exposure_data_1_ok = exposure_data_1.loc[
        ~exposure_data_1['HASH'].isin(bla['name'])
        ]
    exposure_data_1_nonok = exposure_data_1.loc[
        exposure_data_1['HASH'].isin(bla['name'])
        ]

    # TODO: Why just 9010?
    # Majority is 9010 (e.g. duplicates of "M" and "M.from.Perc" cases)
    # Only 9010 treated, remaining cases are deleted
    where_subs_9010 = exposure_data_1_nonok['IMIS_SUBSTANCE_CODE'] == '9010'
    exposure_data_1_nonok_9010 = exposure_data_1_nonok.loc[where_subs_9010]
    exposure_data_1_nonok_9010 = (
        exposure_data_1_nonok_9010.sort_values(by='HASH')
    )

    # One out of 2 sample is retained
    max_rows = len(exposure_data_1_nonok_9010)
    indices = range(0, max_rows, 2)
    exposure_data_1_nonok_9010 = exposure_data_1_nonok_9010.iloc[indices]

    return pd.concat(
        [exposure_data_1_ok, exposure_data_1_nonok_9010]
        )
#endregion

#region: clean_instrument_type
def clean_instrument_type(exposure_data, it_directory, **kwargs):
    '''
    Comprehensive function to handle the cleaning of instrument type.
    '''
    exposure_data = _handle_missing_instrument_type(exposure_data)
    table_for_subs = load_instrument_type_tables(it_directory)
    exposure_data = (
        _apply_instrument_type_tables(exposure_data, table_for_subs)
    )
    exposure_data = _handle_remaining_missing_instrument_type(exposure_data)
    return exposure_data
#endregion

#region: _handle_missing_instrument_type
def _handle_missing_instrument_type(exposure_data):
    '''
    Handle missing instrument type and perform initial population and cleanup.
    '''
    exposure_data = _remove_empty_instrument_type(exposure_data)
    where_nan = exposure_data['INSTRUMENT_TYPE'].isna()
    exposure_data.loc[where_nan, 'INSTRUMENT_TYPE'] = ''

    exposure_data['INSTRUMENT_TYPE_2'] = 'not recorded'  # initialize

    # Copy raw instrument type for 1984-2011
    where_1984_2011 = exposure_data['YEAR'].astype(int) < 2012
    exposure_data.loc[where_1984_2011, 'INSTRUMENT_TYPE_2'] = (
        exposure_data.loc[where_1984_2011, 'INSTRUMENT_TYPE']
    )

    return exposure_data
#endregion

#region: _remove_empty_instrument_type
def _remove_empty_instrument_type(exposure_data):
    '''
    Remove samples where instrument type is an empty string.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = (
        (exposure_data['INSTRUMENT_TYPE'] == '') 
        & exposure_data['INSTRUMENT_TYPE'].notna()
    )
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: load_instrument_type_tables
def load_instrument_type_tables(it_directory):
    '''
    Load IT tables for each substance code.
    '''
    csv_files = [f for f in os.listdir(it_directory) if f.endswith('.csv')]
    table_for_subs = {}

    for file in csv_files:
        # Extract the substance code from the filename
        subs_code = file[2:6]
        # Missing values are replaced with '' like R's read.csv
        df = pd.read_csv(os.path.join(it_directory, file), sep=',').fillna('')
        table_for_subs[subs_code] = df

    return table_for_subs
#endregion

#region: _apply_instrument_type_tables
def _apply_instrument_type_tables(exposure_data, table_for_subs):
    '''
    Clean instrument type for specific substance codes using conversion
    tables.
    '''
    exposure_data = exposure_data.copy()

    for subs_code, it_table in table_for_subs.items():
        # For each clean values, get the corresponding raw value(s)
        for clean_value in it_table['clean'].unique():
            where_clean_value = it_table['clean'] == clean_value
            raw_values_to_clean = list(
                it_table.loc[where_clean_value, 'raw'].astype('str')
                )
            
            where_to_clean = (
                (exposure_data['IMIS_SUBSTANCE_CODE'] == subs_code) 
                & (exposure_data['YEAR'].astype(int) < 2010)
                & (exposure_data['INSTRUMENT_TYPE'].isin(raw_values_to_clean))
                )
            exposure_data.loc[where_to_clean, 'INSTRUMENT_TYPE_2'] = (
                clean_value
            )

    return exposure_data
#endregion

#region: _handle_remaining_missing_instrument_type
def _handle_remaining_missing_instrument_type(exposure_data):
    '''
    Final cleanup for 'INSTRUMENT_TYPE_2'.

    Sets empty strings to 'eliminate' and removes all samples designated as 
    'eliminate', including those set through conversion tables.
    '''
    exposure_data = exposure_data.copy()
    where_empty = exposure_data['INSTRUMENT_TYPE_2'] == ''
    exposure_data.loc[where_empty, 'INSTRUMENT_TYPE_2'] = 'eliminate'
    rows_to_exclude = exposure_data['INSTRUMENT_TYPE_2'] == 'eliminate'
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_zero_volume_sampled
def remove_zero_volume_sampled(exposure_data, **kwargs):
    '''
    Remove samples that have an air volume sampled of zero.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = exposure_data['AIR_VOLUME_SAMPLED'] == 0.
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_missing_volume
def remove_missing_volume(exposure_data, **kwargs):
    '''
    Remove samples that have a missing or empty volume sampled variable.

    This function identifies and removes samples where the 'AIR_VOLUME_SAMPLED'
    column is either missing (NaN) or an empty string ('').
    '''
    exposure_data = exposure_data.copy()
    
    rows_to_exclude = (
        exposure_data['AIR_VOLUME_SAMPLED'].isna() 
        | (exposure_data['AIR_VOLUME_SAMPLED'] == '')
    )

    return exposure_data.loc[~rows_to_exclude]
#endregion

# NOTE: Inconsistency
#region: remove_missing_sample_number
def remove_missing_sample_number(exposure_data, **kwargs):
    '''
    Remove samples that have a missing or null sampling number.
    
    Note:
    - In the original R script, '0' and '0.0' were treated as distinct values 
      because they were stored as strings. However, in Python, when converting 
      to numeric , both '0' and '0.0' are treated as numeric zero (0.0) and 
      thus identified as null values by this function.
    '''
    exposure_data = exposure_data.copy()
    
    rows_to_exclude = (
        exposure_data['SAMPLING_NUMBER'].isna() 
        | (pd.to_numeric(
            exposure_data['SAMPLING_NUMBER'], errors='coerce') == 0.
            )
    )

    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_negative_sample_result
def remove_negative_sample_result(exposure_data, **kwargs):
    '''
    Remove samples with a sample result less than zero.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = exposure_data['SAMPLE_RESULT_3'] < 0.
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_null_time_sampled
def remove_null_time_sampled(exposure_data, **kwargs):
    '''
    Remove samples that have a null time sampled variable.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = exposure_data['TIME_SAMPLED'] == 0.
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_missing_time_sampled
def remove_missing_time_sampled(exposure_data, **kwargs):
    '''
    Remove samples that have a missing value for the time sampled variable.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = exposure_data['TIME_SAMPLED'].isna()
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_missing_office_identifier
def remove_missing_office_identifier(exposure_data, **kwargs):
    '''
    Remove samples that have a missing value for the office ID.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = exposure_data['OFFICE_ID'].isna()
    return exposure_data.loc[~rows_to_exclude]
#endregion

# FIXME: Double check conversion factor. Unclear.
#region: convert_percent_to_mass_concentration
def convert_percent_to_mass_concentration(exposure_data, **kwargs):
    '''
    Convert sample results from percentage concentration to mass concentration 
    (mg/m³).
    '''
    exposure_data = exposure_data.copy()

    exposure_data = remove_null_weight(exposure_data)

    where_to_convert = (
        (exposure_data['SAMPLE_WEIGHT_2'] != 0) 
        & (exposure_data['UNIT_OF_MEASUREMENT_2'] == '%') 
        & (exposure_data['SAMPLE_RESULT_2'] > 0) 
        & exposure_data['SAMPLE_WEIGHT_2'].notna() 
        & exposure_data['AIR_VOLUME_SAMPLED'].notna() 
        & (exposure_data['AIR_VOLUME_SAMPLED'] > 0)
    )

    sample_result = exposure_data.loc[where_to_convert, 'SAMPLE_RESULT_2']
    sample_weight = exposure_data.loc[where_to_convert, 'SAMPLE_WEIGHT_2']
    air_volume_sampled = (
        exposure_data.loc[where_to_convert, 'AIR_VOLUME_SAMPLED']
    )

    conversion_factor = 10.

    converted_result = (
        (sample_result * sample_weight * conversion_factor) 
        / air_volume_sampled
    )

    # Assign the converted results back to the dataframe
    exposure_data['SAMPLE_RESULT_3'] = exposure_data['SAMPLE_RESULT_2']
    exposure_data.loc[where_to_convert, 'SAMPLE_RESULT_3'] = converted_result

    exposure_data.loc[where_to_convert, 'UNIT_OF_MEASUREMENT_2'] = (
        'M.from.Perc'
        )

    return exposure_data
#endregion

#region: remove_null_weight
def remove_null_weight(exposure_data):
    '''
    Remove samples where unit of measurement is percentage ('%'), the sample 
    result is non-null, but the sample weight is null.
    '''
    exposure_data = exposure_data.copy()

    # TODO: Is this step necessary?
    exposure_data['SAMPLE_WEIGHT_2'] = (
        exposure_data['SAMPLE_WEIGHT'].fillna(0)
    )

    rows_to_exclude = (
        (exposure_data['SAMPLE_WEIGHT_2'] == 0) &
        (exposure_data['UNIT_OF_MEASUREMENT_2'] == '%') &
        (exposure_data['SAMPLE_RESULT_2'] > 0)
    )

    return exposure_data.loc[~rows_to_exclude]
#endregion

# TODO: Remove hardcoding?
#region: remove_invalid_unit
def remove_invalid_unit(exposure_data, **kwargs):
    '''
    For each list of substance codes, remove samples where the unit of
    measurement is invalid.
    '''
    top_substances = [
        '0040', '0230', '0260', '0360', '0430', '0491', '0685', 
        '0720', '0731', '1073', '1290', '1520', '1560', '1591', 
        '1620', '1730', '1790', '1840', '2270', '2280', '2460', 
        '2571', '2590', '2610', '9020', '9130', '9135', 'C141', 'S103'
    ]
    valid_units_n31 = ['', 'F', 'P', 'M']
    exposure_data = _remove_invalid_unit_for_substance_codes(
        exposure_data, 
        top_substances, 
        valid_units_n31
        )

    valid_units_n32 = ['', '%', 'M']
    exposure_data = _remove_invalid_unit_for_substance_codes(
        exposure_data, 
        ['9010'], 
        valid_units_n32
        )

    where_other = (
        ~exposure_data['IMIS_SUBSTANCE_CODE'].isin(top_substances + ['9010'])
        )
    other_substances = list(
        exposure_data.loc[where_other, 'IMIS_SUBSTANCE_CODE'].unique()
        )
    valid_units_n33 = ['', '%', 'M', 'P', 'F']
    exposure_data = _remove_invalid_unit_for_substance_codes(
        exposure_data, 
        other_substances, 
        valid_units_n33
        )

    return exposure_data
#endregion

#region: _remove_invalid_unit_for_substance_codes
def _remove_invalid_unit_for_substance_codes(
        exposure_data, 
        substance_codes, 
        valid_units
        ):
    '''
    Remove samples where the unit of measurement is invalid for given
    substances.
    '''
    exposure_data = exposure_data.copy()

    where_in_substance_codes = (
        exposure_data['IMIS_SUBSTANCE_CODE'].isin(substance_codes)
    )
    where_invalid_units = (
        ~exposure_data['UNIT_OF_MEASUREMENT_2'].isin(valid_units)
    )

    rows_to_exclude = where_in_substance_codes & where_invalid_units
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: create_detection_indicator
def create_detection_indicator(exposure_data, **kwargs):
    '''
    Create a new column 'QUALIFIER_2' to indicate detection status.
    '''
    exposure_data = exposure_data.copy()

    exposure_data['QUALIFIER_2'] = 'detected'  # initialize
    where_null = exposure_data['SAMPLE_RESULT_2'] == 0
    exposure_data.loc[where_null, 'QUALIFIER_2'] = 'ND'

    return exposure_data
#endregion

#region: remove_percent_greater_than_100
def remove_percent_greater_than_100(exposure_data, **kwargs):
    '''
    Remove samples where the unit of measurement is '%' and the sample result
    is greater than 100.
    '''
    rows_to_exclude = (
        (exposure_data['UNIT_OF_MEASUREMENT_2'] == '%') &
        (exposure_data['SAMPLE_RESULT_2'] > 100.)
    )
    
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_empty_unit_non_null_result
def remove_empty_unit_non_null_result(exposure_data, **kwargs):
    '''
    Remove samples where the unit of measurement is empty and the sample 
    result is not null.
    '''
    rows_to_exclude = (
        (exposure_data['UNIT_OF_MEASUREMENT_2'] == '') &
        (exposure_data['SAMPLE_RESULT_2'] > 0)
    )
    
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_invalid_fibers_unit
def remove_invalid_fibers_unit(exposure_data, **kwargs):
    '''
    Remove samples with specific substance codes that should not have "F" as
    the unit of measurement.
    '''
    # These codes should not have "F" as the unit of measurement
    invalid_substance_codes = ['1073', '2270', '2470', '9135']
    
    where_invalid_units = (
        (exposure_data['UNIT_OF_MEASUREMENT_2'] == 'F') &
        (exposure_data['IMIS_SUBSTANCE_CODE'].isin(invalid_substance_codes))
    )
    
    return exposure_data.loc[~where_invalid_units]
#endregion

#region: remove_qualifier_unit_mismatch
def remove_qualifier_unit_mismatch(exposure_data, **kwargs):
    '''
    Remove samples with inconsistent qualifier and unit of measurement.
    '''
    exposure_data = exposure_data.copy()

    condition_inconsistent_units = (
        (exposure_data['UNIT_OF_MEASUREMENT_2'] != '%') 
        & (exposure_data['QUALIFIER'] == '%')
    ) | (
        (exposure_data['UNIT_OF_MEASUREMENT_2'] != 'M') 
        & (exposure_data['QUALIFIER'] == 'M')
    )

    return exposure_data.loc[~condition_inconsistent_units]
#endregion

#region: remove_approximate_measure
def remove_approximate_measure(exposure_data, **kwargs):
    '''
    Remove samples where the QUALIFIER indicates an approximate measure.
    '''
    exposure_data = exposure_data.copy()
    
    approximate_qualifiers = [
        '@', 
        ' @', 
        '@<', 
        '@=<', 
        '@<=', 
        '<@', 
        '=<@', 
        'EST'
        ]
    rows_to_exclude = exposure_data['QUALIFIER'].isin(approximate_qualifiers)
    
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_yttrium_substance_conflict
def remove_yttrium_substance_conflict(exposure_data, **kwargs):
    '''
    Remove samples where the qualifier 'Y' is used but the substance code is
    not 9135.
    '''
    exposure_data = exposure_data.copy()

    rows_to_exclude = (
        (exposure_data['QUALIFIER'] == 'Y') 
        & (exposure_data['IMIS_SUBSTANCE_CODE'] != '9135')
    )

    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_fibers_substance_conflict
def remove_fibers_substance_conflict(exposure_data, **kwargs):
    '''
    Remove samples where the qualifier suggests fibers (F) but the substance
    code is not 9020.
    '''
    exposure_data = exposure_data.copy()
    rows_to_exclude = (
        (exposure_data['QUALIFIER'] == 'F') 
        & (exposure_data['IMIS_SUBSTANCE_CODE'] != '9020')
    )
    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_combustion_related
def remove_combustion_related(exposure_data, **kwargs):
    '''
    Remove samples with qualifiers related to combustion.
    '''
    exposure_data = exposure_data.copy()

    combustion_qualifiers = ['COMB', 'COMD', 'com', 'comb']
    rows_to_exclude = exposure_data['QUALIFIER'].isin(combustion_qualifiers)

    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_blk_possible_bulk_not_blank
def remove_blk_possible_bulk_not_blank(
        exposure_data, 
        qualif_conv_2020, 
        **kwargs
        ):
    '''
    Remove samples judged to be possible blank (BLK) and bulk, yet BLANK_USED
    is 'N'.
    '''
    exposure_data = exposure_data.copy()

    condition_blk_possible_bulk = (
        (qualif_conv_2020['clean'] == 'BLK')
        & (qualif_conv_2020['possible_bulk'] == 'Y')
    )
    rows_to_exclude = _rows_to_exclude_based_on_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        condition_blk_possible_bulk
        )

    # Further filter rows where BLANK_USED is 'N'
    rows_to_exclude = rows_to_exclude & (exposure_data['BLANK_USED'] == 'N')

    return exposure_data.loc[~rows_to_exclude]
#endregion

#region: remove_conflicting_qualifier
def remove_conflicting_qualifier(exposure_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples with qualifiers conflicting with sample type.
    '''
    exposure_data = exposure_data.copy()

    where_conflict = qualif_conv_2020['clean'].isin(['B', 'W'])
    exposure_data = _remove_based_on_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        where_conflict
        )
    return exposure_data
#endregion

#region: remove_uninterpretable_qualifier
def remove_uninterpretable_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        **kwargs
        ):
    '''
    Remove samples with qualifiers deemed uninterpretable.
    '''
    exposure_data = exposure_data.copy()

    where_eliminate = qualif_conv_2020['clean'] == 'eliminate'
    exposure_data = _remove_based_on_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        where_eliminate
        )
    return exposure_data
#endregion

#region: remove_blk_not_bulk
def remove_blk_not_bulk(exposure_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples where QUALIFIER is 'BLK' and not possible bulk.
    '''
    exposure_data = exposure_data.copy()

    where_blk_not_bulk  = (
        (qualif_conv_2020['clean'] == 'BLK') 
        & (qualif_conv_2020['possible_bulk'] == 'N')
    )
    exposure_data = _remove_based_on_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        where_blk_not_bulk
        )
    return exposure_data
#endregion

#region: _remove_based_on_qualifier
def _remove_based_on_qualifier(exposure_data, qualif_conv_2020, condition):
    '''
    General function to remove samples based on QUALIFIER conditions.
    '''
    exposure_data = exposure_data.copy()

    rows_to_exclude = _rows_to_exclude_based_on_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        condition
        )
    return exposure_data.loc[~rows_to_exclude]
#endregion:

#region: _rows_to_exclude_based_on_qualifier
def _rows_to_exclude_based_on_qualifier(
        exposure_data, 
        qualif_conv_2020, 
        condition
        ):
    '''
    General function to remove samples based on QUALIFIER conditions.
    '''
    exposure_data = exposure_data.copy()
    
    raw_values_to_exclude = qualif_conv_2020.loc[condition, 'raw']
    return exposure_data['QUALIFIER'].isin(raw_values_to_exclude)
#endregion

#region: clean_unit_of_measurement
def clean_unit_of_measurement(exposure_data, unit_conv_2020, **kwargs):
    '''
    Clean the `UNIT_OF_MEASUREMENT` column by mapping raw values to clean 
    values.
    '''
    exposure_data = exposure_data.copy()

    # Initialize a cleaned column
    exposure_data['UNIT_OF_MEASUREMENT_2'] = (
        exposure_data['UNIT_OF_MEASUREMENT']
    )
    
    for clean_value in unit_conv_2020['clean'].unique():
        where_clean_value = unit_conv_2020['clean'] == clean_value
        raw_values = list(unit_conv_2020.loc[where_clean_value, 'raw'])
        where_needs_clean = (
            exposure_data['UNIT_OF_MEASUREMENT'].isin(raw_values)
        )
        exposure_data.loc[where_needs_clean, 'UNIT_OF_MEASUREMENT_2'] = (
            clean_value
        )

    return exposure_data
#endregion

#region: remove_invalid_non_detect
def remove_invalid_non_detect(exposure_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples where `QUALIFIER` suggests ND but `SAMPLE_RESULT_2` > 0
    and not censored (N08), and where `QUALIFIER` suggests ND or is censored
    but `SAMPLE_RESULT_2` > 0 (N29).
    '''
    exposure_data = exposure_data.copy()

    where_nd = qualif_conv_2020['clean'] == 'ND'
    nd_qualifiers = qualif_conv_2020.loc[where_nd, 'raw']
    condition_n08 = (
        (exposure_data['SAMPLE_RESULT_2'] > 0) 
        & (exposure_data['CENSORED'] != 'Y') 
        & (exposure_data['QUALIFIER'].isin(nd_qualifiers))
    )
    exposure_data = exposure_data.loc[~condition_n08]  # N08

    condition_n29 = (
        (exposure_data['SAMPLE_RESULT_2'] > 0) 
        & ((exposure_data['CENSORED'] == 'Y') 
           | (exposure_data['QUALIFIER'].isin(nd_qualifiers)))
    )
    
    exposure_data = exposure_data.loc[~condition_n29]  # N29

    return exposure_data
#endregion

#region: add_censored_column
def add_censored_column(exposure_data, **kwargs):
    '''
    Add a column indicating that the sample is censored ONLY based on the 
    'QUALIFIER' column.
    '''
    exposure_data = exposure_data.copy()

    new_column = 'CENSORED'
    exposure_data[new_column] = 'N'  # initialize
    qualifier_censored_values = [
        '-<', 
        '  <', 
        ' =<', 
        '@<', 
        '@<=', 
        '@=<', 
        '<', 
        '< =', 
        '<@', 
        '<=', 
        '<= 0', 
        '= <', 
        '=<', 
        '=<@'
    ]
    where_censored = (
        exposure_data['QUALIFIER'].isin(qualifier_censored_values)
    )
    exposure_data.loc[where_censored, new_column] = 'Y'

    # FIXME: Something seems odd. Replacing NA with 'raw was NA' and then ''
    exposure_data['QUALIFIER'] = (
        exposure_data['QUALIFIER'].replace('raw was NA', '')
    )
    exposure_data['SAMPLE_RESULT_2'] = (
        exposure_data['SAMPLE_RESULT'].fillna(0)
    )

    return exposure_data
#endregion

#region: replace_missing_values
def replace_missing_values(exposure_data, **kwargs):
    '''
    '''
    exposure_data = exposure_data.copy()

    for column in ['QUALIFIER', 'UNIT_OF_MEASUREMENT']:
        exposure_data[column] = exposure_data[column].fillna('raw was NA')

    return exposure_data
#endregion

#region: load_unit_measure_conversion
def load_unit_measure_conversion(unit_conv_file):
    '''
    Load conversion table for the 'UNIT_OF_MEASUREMENT' column.
    '''
    unit_conv_2020 = pd.read_csv(unit_conv_file, sep=';')
    unit_conv_2020['clean'] = as_character(unit_conv_2020['clean'])
    unit_conv_2020['raw'] = as_character(unit_conv_2020['raw'])
    return unit_conv_2020
#endregion

#region: load_qualifier_conversion
def load_qualifier_conversion(qualif_conv_file):
    '''
    Load conversion table for the 'QUALIFIER' column.
    '''
    qualif_conv_2020 = pd.read_csv(qualif_conv_file, sep=';')
    qualif_conv_2020['clean'] = as_character(qualif_conv_2020['clean'])
    qualif_conv_2020['raw'] = as_character(qualif_conv_2020['raw'])
    qualif_conv_2020['possible_bulk'] = as_character(
        qualif_conv_2020['possible_bulk']
        )
    return qualif_conv_2020
#endregion

#region: remove_rare_or_non_chemical
def remove_rare_or_non_chemical(exposure_data, **kwargs):
    '''
    Exclude substances with few samples or non-chemical IMIS codes.
    '''
    exposure_data = exposure_data.copy()

    ## Exclude substances with few samples
    subst = exposure_data['IMIS_SUBSTANCE_CODE'].value_counts().reset_index()
    subst.columns = ['code', 'n']
    where_enough = subst['n'] >= 100
    subst = subst[where_enough]

    ## Remove non-chemical substance codes
    # FIXME: Remove hardcoding
    non_chemical_codes = [
        'G301', 'G302', 'Q115', 'T110', 'M125', 'Q116', 'Q100', 'S325'
        ]
    where_non_chemical = subst['code'].isin(non_chemical_codes)
    subst = subst[~where_non_chemical]

    sub_list_all = list(subst['code'])
    rows_to_include = exposure_data['IMIS_SUBSTANCE_CODE'].isin(sub_list_all)
    return exposure_data[rows_to_include]
#endregion

#region: remove_nonpersonal
def remove_nonpersonal(exposure_data, **kwargs):
    '''
    Exclude all samples that are not designated as 'P'.
    '''
    exposure_data = exposure_data.copy()
    not_blank = exposure_data['SAMPLE_TYPE'] != 'P'
    return exposure_data.loc[~not_blank]
#endregion

#region: remove_blanks
def remove_blanks(exposure_data, **kwargs):
    '''
    Remove blanks from the 'BLANK_USED' variable 
    
    Other blanks identified later by 'QUALIFIER'.
    '''
    exposure_data = exposure_data.copy()
    not_blank = exposure_data['BLANK_USED'] == 'N'
    return exposure_data.loc[not_blank]
#endregion

# TODO: Double check whether these are all relevant
#region: pre_clean
def pre_clean(exposure_data, **kwargs):
    '''
    '''
    exposure_data = exposure_data.copy()
        
    exposure_data['AIR_VOLUME_SAMPLED'] = pd.to_numeric(
        exposure_data['AIR_VOLUME_SAMPLED'], errors='coerce'
        )

    exposure_data['BLANK_USED'] = factor(
        exposure_data['BLANK_USED'], categories=['Y', 'N']
        )

    exposure_data['CITY'] = as_character(exposure_data['CITY'])

    exposure_data['DATE_REPORTED'] = (
        convert_date(exposure_data['DATE_REPORTED'])
    )
    exposure_data['DATE_SAMPLED'] = (
        convert_date(exposure_data['DATE_SAMPLED'])
    )

    exposure_data['EIGHT_HOUR_TWA_CALC'] = factor(
        exposure_data['EIGHT_HOUR_TWA_CALC'], categories=['Y', 'N']
        )

    exposure_data['ESTABLISHMENT_NAME'] = as_character(
        exposure_data['ESTABLISHMENT_NAME']
        )
    exposure_data['FIELD_NUMBER'] = (
        as_character(exposure_data['FIELD_NUMBER'])
    )

    # NOTE: Seems unnecessary to go from one type to another
    exposure_data['IMIS_SUBSTANCE_CODE'] = factor(
        exposure_data['IMIS_SUBSTANCE_CODE'].str.replace(' ', '0').str.zfill(4)
    )
    exposure_data['IMIS_SUBSTANCE_CODE'] = as_character(
        exposure_data['IMIS_SUBSTANCE_CODE']
        )
    
    exposure_data['INSPECTION_NUMBER'] = (
        factor(exposure_data['INSPECTION_NUMBER'])
    )
    exposure_data['INSPECTION_NUMBER'] = (
        as_character(exposure_data['INSPECTION_NUMBER'])
    )

    exposure_data['INSTRUMENT_TYPE'] = (
        as_character(exposure_data['INSTRUMENT_TYPE'])
    )
    exposure_data['LAB_NUMBER'] = factor(exposure_data['LAB_NUMBER'])

    exposure_data['NAICS_CODE'] = (
        as_character(exposure_data['NAICS_CODE'])
        .apply(
            lambda x: x if isinstance(x, str) and len(x) >= 6 else np.nan)
    )

    exposure_data['OFFICE_ID'] = factor(exposure_data['OFFICE_ID'])
    exposure_data['QUALIFIER'] = as_character(exposure_data['QUALIFIER'])

    exposure_data['SAMPLE_RESULT'] = pd.to_numeric(
        exposure_data['SAMPLE_RESULT'], errors='coerce'
        )

    exposure_data['SAMPLE_TYPE'] = factor(exposure_data['SAMPLE_TYPE'])
    exposure_data['SAMPLE_WEIGHT'] = pd.to_numeric(
        exposure_data['SAMPLE_WEIGHT'], errors='coerce'
        )
    exposure_data['SAMPLING_NUMBER'] = (
        factor(exposure_data['SAMPLING_NUMBER'])
    )
    exposure_data['SAMPLING_NUMBER'] = (
        as_character(exposure_data['SAMPLING_NUMBER'])
    )

    exposure_data['SIC_CODE'] = factor(exposure_data['SIC_CODE'])
    exposure_data['STATE'] = factor(exposure_data['STATE'])
    exposure_data['SUBSTANCE'] = as_character(exposure_data['SUBSTANCE'])

    exposure_data['TIME_SAMPLED'] = pd.to_numeric(
        exposure_data['TIME_SAMPLED'], errors='coerce'
        )
    exposure_data['UNIT_OF_MEASUREMENT'] = as_character(
        exposure_data['UNIT_OF_MEASUREMENT']
        )

    exposure_data['ZIP_CODE'] = (
        as_character(exposure_data['ZIP_CODE'])
        .str.replace(' ', '0').str.zfill(5)
    )
    exposure_data['ZIP_CODE'] = factor(exposure_data['ZIP_CODE'])

    exposure_data['YEAR'] = factor(exposure_data['DATE_SAMPLED'].dt.year)

    exposure_data['INSPECTION_NUMBER'] = (
        exposure_data['INSPECTION_NUMBER'].str.strip()
    )
    exposure_data['SAMPLING_NUMBER'] = (
        exposure_data['SAMPLING_NUMBER'].str.strip()
    )

    return exposure_data
#endregion

#region: as_character
def as_character(column):
    '''
    Mimic R's as.character function in Python. 
    
    This may not account for all differences.
    '''
    return column.apply(lambda x : str(x) if pd.notna(x) else np.nan)
#endregion

#region: factor
def factor(series, categories=None):
    '''
    Mimic R's factor function in Python using pandas.

    This may not account for all differences.
    '''    
    cat_series = pd.Categorical(series, categories=categories, ordered=True)
    return cat_series
#endregion

#region: convert_date
def convert_date(column):
    '''
    Lowercase and date conversion handling multiple formats
    '''
    return pd.to_datetime(
        column.str.lower(),
        errors='coerce',
        format='%Y-%b-%d'
    ).combine_first(
        pd.to_datetime(column.str.lower(), errors='coerce', format='%Y/%m/%d')
    ).combine_first(
        pd.to_datetime(column.str.lower(), errors='coerce', format='%Y-%m-%d')
    )
#endregion

#region: compare_columns
def compare_columns(df1, df2):
    '''
    Helper function to check for discrepancies between dataframes.
    '''
    discrepancies = {}

    if any(df1.columns != df2.columns):
        raise ValueError('The columns are not identical')
    if any(df1.index != df2.index):
        raise ValueError('The indexes are not identical')
    
    for col in df1.columns.intersection(df2.columns):

        # Convert to str to avoid issues related to dtypes
        col1, col2 = df1[col].astype('str'), df2[col].astype('str')

        where_both_not_nan = ~(col1.isna() & col2.isna())
        where_discrepancy = (col1 != col2) & where_both_not_nan

        if any(where_discrepancy):
            discrepancies[col] = (
                col1.loc[where_discrepancy], 
                col2.loc[where_discrepancy]
                )

    return discrepancies
#endregion

#region: plot_cumulative_changes
def plot_cumulative_changes(log_file, initial_count):
    '''
    Plot the cumulative proportion of samples remaining after each cleaning 
    step.
    '''
    change_log = load_change_log(log_file)

    cumulative_pairs = prepare_cumulative_data(change_log, initial_count)
    steps = [step for step, _ in cumulative_pairs]
    cum_values = [count for _, count in cumulative_pairs]

    # Reverse the data to have the "full dataset" at the top
    steps = steps[::-1]
    cum_values = cum_values[::-1]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(cum_values, steps, marker='o', linestyle='-')
    ax.set_title('Proportion of Samples Remaining After Each Cleaning Step')
    ax.set_xlabel('Proportion Remaining (%)')
    ax.set_ylabel('Cleaning Step')
    ax.grid(True)
    
    # Invert the x-axis so that it decreases from left to right
    plt.gca().invert_xaxis()

    return fig, ax
#endregion

#region: load_change_log
def load_change_log(log_file):
    '''
    Load the sample size change log from a JSON file.

    Returns
    -------
    dict
    '''
    with open(log_file, 'r') as file:
        change_log = json.load(file)
    return change_log
#endregion

#region: prepare_cumulative_data
def prepare_cumulative_data(change_log, initial_count):
    '''
    Prepare the data for plotting the cumulative proportion of samples 
    remaining after each cleaning step.
    '''
    # Initialize with the full dataset
    cum_count = initial_count
    cumulative_pairs = [(f'0. Full dataset ({cum_count:,})', cum_count)]
    
    def reformat_key(k, step_number):
        return f"{step_number}. {k.capitalize().replace('_', ' ')}"
    
    for step_number, (k, v) in enumerate(change_log.items(), start=1):
        if abs(v) > 0:
            formatted_key = reformat_key(k, step_number)
            # Include the cumulative count of remaining samples
            cum_count += v  # where v = N_after - N_before
            cumulative_pairs.append((formatted_key, cum_count))

    # Convert the counts to proportions
    TO_PERCENT = 100
    cumulative_pairs = [
        (k, v/initial_count*TO_PERCENT) 
        for k, v in cumulative_pairs
        ]

    return cumulative_pairs
#endregion