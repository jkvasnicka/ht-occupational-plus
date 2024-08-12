'''
This module implements a data cleaning methodology by Jérôme Lavoué - 
Université de Montréal for the Chemical Exposure Health Data (CEHD).

The original R script was translated to Python.
'''
import pandas as pd
import numpy as np
import os

CLEANING_STEPS = [
    'pre_clean',
    'remove_blanks',
    'remove_nonpersonal',
    'exclude_few',
    'replace_missing_values',
    'add_censored_column',
    'remove_invalid_nd',
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
    'remove_invalid_unit_f',
    'remove_empty_unit_non_null_result',
    'remove_percent_greater_than_100',
    'create_detection_indicator',
    'remove_invalid_unit_for_all_substances',
    'convert_percent_to_mass_concentration',
    'remove_missing_office_id',
    'remove_missing_time_sampled',
    'remove_null_time_sampled',
    'remove_negative_sample_result',
    'remove_missing_sample_number',
    'remove_missing_volume',
    'remove_zero_volume_sampled',
    'clean_instrument_type',
    'clean_duplicates'
]

#region: clean_cehd_data
def clean_cehd_data(cehd_data, path_settings):
    '''
    '''
    # Define key-word arguments for flexible argument passing
    kwargs = path_settings.copy()
    kwargs['qualif_conv_2020'] = load_qualifier_conversion(
        path_settings['qualif_conv_file']
        )
    kwargs['unit_conv_2020'] = load_unit_measure_conversion(
        path_settings['unit_conv_file']
    )

    for step_name in CLEANING_STEPS:
        cehd_data = _apply_cleaning_step(cehd_data, step_name, kwargs)
    
    return cehd_data
#endregion

#region: _apply_cleaning_step
def _apply_cleaning_step(cehd_data, step_name, kwargs):
    '''
    '''
    return globals()[step_name](cehd_data, **kwargs)
#endregion

#region: clean_duplicates
def clean_duplicates(cehd_data, **kwargs):
    '''
    Clean the dataset by identifying and removing duplicate samples.
    '''
    cehd_data = _create_hash(cehd_data)
    bla = _identify_potential_duplicates(cehd_data)
    false_duplicate_hashes = _identify_false_duplicates(cehd_data, bla)
    return _remove_true_duplicates(cehd_data, false_duplicate_hashes, bla)
#endregion

#region: _create_hash
def _create_hash(cehd_data):
    '''
    Create a unique HASH variable to identify potential duplicates.
    '''
    cehd_data = cehd_data.copy()
    cehd_data['HASH'] = (
        cehd_data['INSPECTION_NUMBER'].astype(str) + '-' +
        cehd_data['IMIS_SUBSTANCE_CODE'].astype(str) + '-' +
        cehd_data['SAMPLING_NUMBER'].astype(str) + '-' +
        cehd_data['FIELD_NUMBER'].astype(str)
    )
    return cehd_data
#endregion

#region: _identify_potential_duplicates
def _identify_potential_duplicates(cehd_data):
    '''
    Identify and return a DataFrame of potential duplicate records based on 
    the HASH variable.
    '''
    cehd_data = cehd_data.copy()

    bla = cehd_data['HASH'].value_counts().reset_index()
    bla.columns = ['name', 'n']
    bla = bla[bla['n'] > 1]
    bla['name'] = bla['name'].astype(str)

    # Match the values for 'code' and 'sub'
    bla['code'] = bla['name'].map(
        dict(zip(cehd_data['HASH'], cehd_data['IMIS_SUBSTANCE_CODE']))
    )
    bla['sub'] = bla['name'].map(
        dict(zip(cehd_data['HASH'], cehd_data['SUBSTANCE']))
    )

    # Ensure the order matches
    return bla.sort_values(by='name').reset_index(drop=True)
#endregion

#region: _identify_false_duplicates
def _identify_false_duplicates(cehd_data, bla):
    '''
    Identify false duplicates by comparing additional variables (CONCAT).
    
    False duplicates are identified where the CONCAT variable varies 
    for the same HASH.
    '''
    # Create a new hash variable to identify false duplicates
    cehd_data['CONCAT'] = (
        cehd_data['LAB_NUMBER'].astype(str) + '-' +
        cehd_data['STATE'].astype(str) + '-' +
        cehd_data['ZIP_CODE'].astype(str) + '-' +
        cehd_data['YEAR'].astype(str) + '-' +
        cehd_data['TIME_SAMPLED'].astype(str) + '-' +
        cehd_data['SAMPLE_WEIGHT_2'].astype(str)
    )

    # Identify samples where CONCAT is the same
    concat_counts = cehd_data.groupby('HASH')['CONCAT'].nunique()
    concatdiff_hashes = concat_counts.loc[concat_counts > 1].index
    bla['concatdiff'] = bla['name'].isin(concatdiff_hashes)

    # False duplicates occur where CONCAT varies
    return bla['name'].loc[bla['concatdiff']].to_numpy()
#endregion

# FIXME: The original R code
#region: _remove_true_duplicates
def _remove_true_duplicates(cehd_data, false_duplicate_hashes, bla):
    '''
    Remove true duplicates from the dataset, retaining only one sample per 
    duplicate.

    Notes
    -----
    The original R code hardcoded 'max_rows' to 6083, which resulted in the 
    R code inadvertently creating additional rows with NaN. This Python code
    addresses this issue by correctly defining 'max_rows' based on the length.
    '''
    cehd_data = cehd_data.copy()

    restrictM = cehd_data['HASH'].isin(false_duplicate_hashes)

    cehd_data_1 = cehd_data.loc[~restrictM]

    #### N: true duplicates ####
    # Separate the DB into the OK and remaining problematic
    cehd_data_1_ok = cehd_data_1.loc[~cehd_data_1['HASH'].isin(bla['name'])]
    cehd_data_1_nonok = cehd_data_1.loc[cehd_data_1['HASH'].isin(bla['name'])]

    # TODO: Why just 9010?
    # Majority is 9010 (e.g. duplicates of "M" and "M.from.Perc" cases)
    # Only 9010 treated, remaining cases are deleted
    where_subs_9010 = cehd_data_1_nonok['IMIS_SUBSTANCE_CODE'] == '9010'
    cehd_data_1_nonok_9010 = cehd_data_1_nonok.loc[where_subs_9010]
    cehd_data_1_nonok_9010 = cehd_data_1_nonok_9010.sort_values(by='HASH')

    # TODO: Why keep every second sample?
    # One out of 2 sample is retained
    max_rows = len(cehd_data_1_nonok_9010)
    indices = range(0, max_rows, 2)
    cehd_data_1_nonok_9010 = cehd_data_1_nonok_9010.iloc[indices]

    return pd.concat(
        [cehd_data_1_ok, cehd_data_1_nonok_9010], 
        ignore_index=True
        )
#endregion

#region: clean_instrument_type
def clean_instrument_type(cehd_data, it_directory, **kwargs):
    '''
    Comprehensive function to handle the cleaning of instrument type.
    '''
    cehd_data = _handle_missing_instrument_type(cehd_data)
    table_for_subs = load_instrument_type_tables(it_directory)
    cehd_data = _apply_instrument_type_tables(cehd_data, table_for_subs)
    cehd_data = _handle_remaining_missing_instrument_type(cehd_data)
    return cehd_data
#endregion

#region: _handle_missing_instrument_type
def _handle_missing_instrument_type(cehd_data):
    '''
    Handle missing instrument type and perform initial population and cleanup.
    '''
    cehd_data = _remove_empty_instrument_type(cehd_data)
    where_nan = cehd_data['INSTRUMENT_TYPE'].isna()
    cehd_data.loc[where_nan, 'INSTRUMENT_TYPE'] = ''

    cehd_data['INSTRUMENT_TYPE_2'] = 'not recorded'  # initialize

    # Copy raw instrument type for 1984-2011
    where_1984_2011 = cehd_data['YEAR'].astype(int) < 2012
    cehd_data.loc[where_1984_2011, 'INSTRUMENT_TYPE_2'] = (
        cehd_data.loc[where_1984_2011, 'INSTRUMENT_TYPE']
    )

    return cehd_data
#endregion

#region: _remove_empty_instrument_type
def _remove_empty_instrument_type(cehd_data):
    '''
    Remove samples where instrument type is an empty string.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = (
        (cehd_data['INSTRUMENT_TYPE'] == '') 
        & cehd_data['INSTRUMENT_TYPE'].notna()
    )
    return cehd_data.loc[~rows_to_exclude]
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
def _apply_instrument_type_tables(cehd_data, table_for_subs):
    '''
    Clean instrument type for specific substance codes using conversion
    tables.
    '''
    cehd_data = cehd_data.copy()

    for subs_code, it_table in table_for_subs.items():
        # For each clean values, get the corresponding raw value(s)
        for clean_value in it_table['clean'].unique():
            where_clean_value = it_table['clean'] == clean_value
            raw_values_to_clean = list(
                it_table.loc[where_clean_value, 'raw'].astype('str')
                )
            
            where_to_clean = (
                (cehd_data['IMIS_SUBSTANCE_CODE'] == subs_code) 
                & (cehd_data['YEAR'].astype(int) < 2010)
                & (cehd_data['INSTRUMENT_TYPE'].isin(raw_values_to_clean))
                )
            cehd_data.loc[where_to_clean, 'INSTRUMENT_TYPE_2'] = clean_value

    return cehd_data
#endregion

#region: _handle_remaining_missing_instrument_type
def _handle_remaining_missing_instrument_type(cehd_data):
    '''
    Final cleanup for 'INSTRUMENT_TYPE_2'.

    Sets empty strings to 'eliminate' and removes all samples designated as 
    'eliminate', including those set through conversion tables.
    '''
    cehd_data = cehd_data.copy()
    where_empty = cehd_data['INSTRUMENT_TYPE_2'] == ''
    cehd_data.loc[where_empty, 'INSTRUMENT_TYPE_2'] = 'eliminate'
    rows_to_exclude = cehd_data['INSTRUMENT_TYPE_2'] == 'eliminate'
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_zero_volume_sampled
def remove_zero_volume_sampled(cehd_data, **kwargs):
    '''
    Remove samples that have an air volume sampled of zero.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = cehd_data['AIR_VOLUME_SAMPLED'] == 0.
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_missing_volume
def remove_missing_volume(cehd_data, **kwargs):
    '''
    Remove samples that have a missing or empty volume sampled variable.

    This function identifies and removes samples where the 'AIR_VOLUME_SAMPLED'
    column is either missing (NaN) or an empty string ('').
    '''
    cehd_data = cehd_data.copy()
    
    rows_to_exclude = (
        cehd_data['AIR_VOLUME_SAMPLED'].isna() 
        | (cehd_data['AIR_VOLUME_SAMPLED'] == '')
    )

    return cehd_data.loc[~rows_to_exclude]
#endregion

# NOTE: Inconsistency
#region: remove_missing_sample_number
def remove_missing_sample_number(cehd_data, **kwargs):
    '''
    Remove samples that have a missing or null sampling number.
    
    Note:
    - In the original R script, '0' and '0.0' were treated as distinct values 
      because they were stored as strings. However, in Python, when converting 
      to numeric , both '0' and '0.0' are treated as numeric zero (0.0) and 
      thus identified as null values by this function.
    '''
    cehd_data = cehd_data.copy()
    
    rows_to_exclude = (
        cehd_data['SAMPLING_NUMBER'].isna() 
        | (pd.to_numeric(cehd_data['SAMPLING_NUMBER'], errors='coerce') == 0.)
    )

    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_negative_sample_result
def remove_negative_sample_result(cehd_data, **kwargs):
    '''
    Remove samples with a sample result less than zero.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = cehd_data['SAMPLE_RESULT_3'] < 0.
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_null_time_sampled
def remove_null_time_sampled(cehd_data, **kwargs):
    '''
    Remove samples that have a null time sampled variable.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = cehd_data['TIME_SAMPLED'] == 0.
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_missing_time_sampled
def remove_missing_time_sampled(cehd_data, **kwargs):
    '''
    Remove samples that have a missing value for the time sampled variable.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = cehd_data['TIME_SAMPLED'].isna()
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_missing_office_id
def remove_missing_office_id(cehd_data, **kwargs):
    '''
    Remove samples that have a missing value for the office ID.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = cehd_data['OFFICE_ID'].isna()
    return cehd_data.loc[~rows_to_exclude]
#endregion

# FIXME: Double check conversion factor. Unclear.
#region: convert_percent_to_mass_concentration
def convert_percent_to_mass_concentration(cehd_data, **kwargs):
    '''
    Convert sample results from percentage concentration to mass concentration 
    (mg/m³).
    '''
    cehd_data = cehd_data.copy()

    cehd_data = remove_null_weight(cehd_data)

    where_to_convert = (
        (cehd_data['SAMPLE_WEIGHT_2'] != 0) 
        & (cehd_data['UNIT_OF_MEASUREMENT_2'] == '%') 
        & (cehd_data['SAMPLE_RESULT_2'] > 0) 
        & cehd_data['SAMPLE_WEIGHT_2'].notna() 
        & cehd_data['AIR_VOLUME_SAMPLED'].notna() 
        & (cehd_data['AIR_VOLUME_SAMPLED'] > 0)
    )

    sample_result = cehd_data.loc[where_to_convert, 'SAMPLE_RESULT_2']
    sample_weight = cehd_data.loc[where_to_convert, 'SAMPLE_WEIGHT_2']
    air_volume_sampled = cehd_data.loc[where_to_convert, 'AIR_VOLUME_SAMPLED']

    conversion_factor = 10.

    converted_result = (
        (sample_result * sample_weight * conversion_factor) 
        / air_volume_sampled
    )

    # Assign the converted results back to the dataframe
    cehd_data['SAMPLE_RESULT_3'] = cehd_data['SAMPLE_RESULT_2']
    cehd_data.loc[where_to_convert, 'SAMPLE_RESULT_3'] = converted_result

    cehd_data.loc[where_to_convert, 'UNIT_OF_MEASUREMENT_2'] = 'M.from.Perc'

    return cehd_data
#endregion

#region: remove_null_weight
def remove_null_weight(cehd_data):
    '''
    Remove samples where unit of measurement is percentage ('%'), the sample 
    result is non-null, but the sample weight is null.
    '''
    cehd_data = cehd_data.copy()

    # TODO: Is this step necessary?
    cehd_data['SAMPLE_WEIGHT_2'] = cehd_data['SAMPLE_WEIGHT'].fillna(0)

    rows_to_exclude = (
        (cehd_data['SAMPLE_WEIGHT_2'] == 0) &
        (cehd_data['UNIT_OF_MEASUREMENT_2'] == '%') &
        (cehd_data['SAMPLE_RESULT_2'] > 0)
    )

    return cehd_data.loc[~rows_to_exclude]
#endregion

# TODO: Remove hardcoding?
#region: remove_invalid_unit_for_all_substances
def remove_invalid_unit_for_all_substances(cehd_data, **kwargs):
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
    cehd_data = _remove_invalid_unit_for_substances(
        cehd_data, 
        top_substances, 
        valid_units_n31
        )

    valid_units_n32 = ['', '%', 'M']
    cehd_data = _remove_invalid_unit_for_substances(
        cehd_data, 
        ['9010'], 
        valid_units_n32
        )

    where_other_substances = (
        ~cehd_data['IMIS_SUBSTANCE_CODE'].isin(top_substances + ['9010'])
        )
    other_substances = list(
        cehd_data.loc[where_other_substances, 'IMIS_SUBSTANCE_CODE'].unique()
        )
    valid_units_n33 = ['', '%', 'M', 'P', 'F']
    cehd_data = _remove_invalid_unit_for_substances(
        cehd_data, 
        other_substances, 
        valid_units_n33
        )

    return cehd_data
#endregion

#region: _remove_invalid_unit_for_substances
def _remove_invalid_unit_for_substances(
        cehd_data, 
        substance_codes, 
        valid_units
        ):
    '''
    Remove samples where the unit of measurement is invalid for given
    substances.
    '''
    cehd_data = cehd_data.copy()

    where_in_substance_codes = (
        cehd_data['IMIS_SUBSTANCE_CODE'].isin(substance_codes)
    )
    where_invalid_units = ~cehd_data['UNIT_OF_MEASUREMENT_2'].isin(valid_units)

    rows_to_exclude = where_in_substance_codes & where_invalid_units
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: create_detection_indicator
def create_detection_indicator(cehd_data, **kwargs):
    '''
    Create a new column 'QUALIFIER_2' to indicate detection status.
    '''
    cehd_data = cehd_data.copy()

    cehd_data['QUALIFIER_2'] = 'detected'  # initialize
    cehd_data.loc[cehd_data['SAMPLE_RESULT_2'] == 0, 'QUALIFIER_2'] = 'ND'

    return cehd_data
#endregion

#region: remove_percent_greater_than_100
def remove_percent_greater_than_100(cehd_data, **kwargs):
    '''
    Remove samples where the unit of measurement is '%' and the sample result
    is greater than 100.
    '''
    rows_to_exclude = (
        (cehd_data['UNIT_OF_MEASUREMENT_2'] == '%') &
        (cehd_data['SAMPLE_RESULT_2'] > 100.)
    )
    
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_empty_unit_non_null_result
def remove_empty_unit_non_null_result(cehd_data, **kwargs):
    '''
    Remove samples where the unit of measurement is empty and the sample 
    result is not null.
    '''
    rows_to_exclude = (
        (cehd_data['UNIT_OF_MEASUREMENT_2'] == '') &
        (cehd_data['SAMPLE_RESULT_2'] > 0)
    )
    
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_invalid_unit_f
def remove_invalid_unit_f(cehd_data, **kwargs):
    '''
    Remove samples with specific substance codes that should not have "F" as
    the unit of measurement.
    '''
    # These codes should not have "F" as the unit of measurement
    invalid_substance_codes = ['1073', '2270', '2470', '9135']
    
    where_invalid_units = (
        (cehd_data['UNIT_OF_MEASUREMENT_2'] == 'F') &
        (cehd_data['IMIS_SUBSTANCE_CODE'].isin(invalid_substance_codes))
    )
    
    return cehd_data.loc[~where_invalid_units]
#endregion

#region: remove_qualifier_unit_mismatch
def remove_qualifier_unit_mismatch(cehd_data, **kwargs):
    '''
    Remove samples with inconsistent qualifier and unit of measurement.
    '''
    cehd_data = cehd_data.copy()

    condition_inconsistent_units = (
        (cehd_data['UNIT_OF_MEASUREMENT_2'] != '%') 
        & (cehd_data['QUALIFIER'] == '%')
    ) | (
        (cehd_data['UNIT_OF_MEASUREMENT_2'] != 'M') 
        & (cehd_data['QUALIFIER'] == 'M')
    )

    return cehd_data.loc[~condition_inconsistent_units]
#endregion

#region: remove_approximate_measure
def remove_approximate_measure(cehd_data, **kwargs):
    '''
    Remove samples where the QUALIFIER indicates an approximate measure.
    '''
    cehd_data = cehd_data.copy()
    
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
    rows_to_exclude = cehd_data['QUALIFIER'].isin(approximate_qualifiers)
    
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_yttrium_substance_conflict
def remove_yttrium_substance_conflict(cehd_data, **kwargs):
    '''
    Remove samples where the qualifier 'Y' is used but the substance code is
    not 9135.
    '''
    cehd_data = cehd_data.copy()

    rows_to_exclude = (
        (cehd_data['QUALIFIER'] == 'Y') 
        & (cehd_data['IMIS_SUBSTANCE_CODE'] != '9135')
    )

    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_fibers_substance_conflict
def remove_fibers_substance_conflict(cehd_data, **kwargs):
    '''
    Remove samples where the qualifier suggests fibers (F) but the substance
    code is not 9020.
    '''
    cehd_data = cehd_data.copy()
    rows_to_exclude = (
        (cehd_data['QUALIFIER'] == 'F') 
        & (cehd_data['IMIS_SUBSTANCE_CODE'] != '9020')
    )
    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_combustion_related
def remove_combustion_related(cehd_data, **kwargs):
    '''
    Remove samples with qualifiers related to combustion.
    '''
    cehd_data = cehd_data.copy()

    combustion_qualifiers = ['COMB', 'COMD', 'com', 'comb']
    rows_to_exclude = cehd_data['QUALIFIER'].isin(combustion_qualifiers)

    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_blk_possible_bulk_not_blank
def remove_blk_possible_bulk_not_blank(cehd_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples judged to be possible blank (BLK) and bulk, yet BLANK_USED
    is 'N'.
    '''
    cehd_data = cehd_data.copy()

    condition_blk_possible_bulk = (
        (qualif_conv_2020['clean'] == 'BLK')
        & (qualif_conv_2020['possible_bulk'] == 'Y')
    )
    rows_to_exclude = _rows_to_exclude_based_on_qualifier(
        cehd_data, 
        qualif_conv_2020, 
        condition_blk_possible_bulk
        )

    # Further filter rows where BLANK_USED is 'N'
    rows_to_exclude = rows_to_exclude & (cehd_data['BLANK_USED'] == 'N')

    return cehd_data.loc[~rows_to_exclude]
#endregion

#region: remove_conflicting_qualifier
def remove_conflicting_qualifier(cehd_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples with qualifiers conflicting with sample type.
    '''
    cehd_data = cehd_data.copy()

    where_conflict = qualif_conv_2020['clean'].isin(['B', 'W'])
    cehd_data = _remove_based_on_qualifier(
        cehd_data, 
        qualif_conv_2020, 
        where_conflict
        )
    return cehd_data
#endregion

#region: remove_uninterpretable_qualifier
def remove_uninterpretable_qualifier(cehd_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples with qualifiers deemed uninterpretable.
    '''
    cehd_data = cehd_data.copy()

    where_eliminate = qualif_conv_2020['clean'] == 'eliminate'
    cehd_data = _remove_based_on_qualifier(
        cehd_data, 
        qualif_conv_2020, 
        where_eliminate
        )
    return cehd_data
#endregion

#region: remove_blk_not_bulk
def remove_blk_not_bulk(cehd_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples where QUALIFIER is 'BLK' and not possible bulk.
    '''
    cehd_data = cehd_data.copy()

    where_blk_not_bulk  = (
        (qualif_conv_2020['clean'] == 'BLK') 
        & (qualif_conv_2020['possible_bulk'] == 'N')
    )
    cehd_data = _remove_based_on_qualifier(
        cehd_data, 
        qualif_conv_2020, 
        where_blk_not_bulk
        )
    return cehd_data
#endregion

#region: _remove_based_on_qualifier
def _remove_based_on_qualifier(cehd_data, qualif_conv_2020, condition):
    '''
    General function to remove samples based on QUALIFIER conditions.
    '''
    cehd_data = cehd_data.copy()

    rows_to_exclude = _rows_to_exclude_based_on_qualifier(
        cehd_data, 
        qualif_conv_2020, 
        condition
        )
    return cehd_data.loc[~rows_to_exclude]
#endregion:

#region: _rows_to_exclude_based_on_qualifier
def _rows_to_exclude_based_on_qualifier(cehd_data, qualif_conv_2020, condition):
    '''
    General function to remove samples based on QUALIFIER conditions.
    '''
    cehd_data = cehd_data.copy()
    
    raw_values_to_exclude = qualif_conv_2020.loc[condition, 'raw']
    return cehd_data['QUALIFIER'].isin(raw_values_to_exclude)
#endregion

#region: clean_unit_of_measurement
def clean_unit_of_measurement(cehd_data, unit_conv_2020, **kwargs):
    '''
    Clean the `UNIT_OF_MEASUREMENT` column by mapping raw values to clean 
    values.
    '''
    cehd_data = cehd_data.copy()

    # Initialize a cleaned column
    cehd_data['UNIT_OF_MEASUREMENT_2'] = cehd_data['UNIT_OF_MEASUREMENT']
    
    for clean_value in unit_conv_2020['clean'].unique():
        where_clean_value = unit_conv_2020['clean'] == clean_value
        raw_values = list(unit_conv_2020.loc[where_clean_value, 'raw'])
        where_needs_clean = cehd_data['UNIT_OF_MEASUREMENT'].isin(raw_values)
        cehd_data.loc[where_needs_clean, 'UNIT_OF_MEASUREMENT_2'] = clean_value

    return cehd_data
#endregion

#region: remove_invalid_nd
def remove_invalid_nd(cehd_data, qualif_conv_2020, **kwargs):
    '''
    Remove samples where `QUALIFIER` suggests ND but `SAMPLE_RESULT_2` > 0
    and not censored (N08), and where `QUALIFIER` suggests ND or is censored
    but `SAMPLE_RESULT_2` > 0 (N29).
    '''
    cehd_data = cehd_data.copy()

    where_nd = qualif_conv_2020['clean'] == 'ND'
    nd_qualifiers = qualif_conv_2020.loc[where_nd, 'raw']
    condition_n08 = (
        (cehd_data['SAMPLE_RESULT_2'] > 0) 
        & (cehd_data['CENSORED'] != 'Y') 
        & (cehd_data['QUALIFIER'].isin(nd_qualifiers))
    )
    cehd_data = cehd_data.loc[~condition_n08]  # N08

    condition_n29 = (
        (cehd_data['SAMPLE_RESULT_2'] > 0) 
        & ((cehd_data['CENSORED'] == 'Y') 
           | (cehd_data['QUALIFIER'].isin(nd_qualifiers)))
    )
    
    cehd_data = cehd_data.loc[~condition_n29]  # N29

    return cehd_data
#endregion

#region: add_censored_column
def add_censored_column(cehd_data, **kwargs):
    '''
    Add a column indicating that the sample is censored ONLY based on the 
    'QUALIFIER' column.
    '''
    cehd_data = cehd_data.copy()

    new_column = 'CENSORED'
    cehd_data[new_column] = 'N'  # initialize
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
    where_censored = cehd_data['QUALIFIER'].isin(qualifier_censored_values)
    cehd_data.loc[where_censored, new_column] = 'Y'

    # FIXME: Something seems odd. Replacing NA with 'raw was NA' and then ''
    cehd_data['QUALIFIER'] = cehd_data['QUALIFIER'].replace('raw was NA', '')
    cehd_data['SAMPLE_RESULT_2'] = cehd_data['SAMPLE_RESULT'].fillna(0)

    return cehd_data
#endregion

#region: replace_missing_values
def replace_missing_values(cehd_data, **kwargs):
    '''
    '''
    cehd_data = cehd_data.copy()

    for column in ['QUALIFIER', 'UNIT_OF_MEASUREMENT']:
        cehd_data[column] = cehd_data[column].fillna('raw was NA')

    return cehd_data
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

#region: exclude_few
def exclude_few(cehd_data, **kwargs):
    '''
    Exclude substances with few samples or non-chemical IMIS codes.
    '''
    cehd_data = cehd_data.copy()

    ## Exclude substances with few samples
    subst = cehd_data['IMIS_SUBSTANCE_CODE'].value_counts().reset_index()
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
    return cehd_data[cehd_data['IMIS_SUBSTANCE_CODE'].isin(sub_list_all)]
#endregion

#region: remove_nonpersonal
def remove_nonpersonal(cehd_data, **kwargs):
    '''
    Exclude all samples that are not designated as 'P'.
    '''
    cehd_data = cehd_data.copy()
    not_blank = cehd_data['SAMPLE_TYPE'] != 'P'
    return cehd_data.loc[~not_blank]
#endregion

#region: remove_blanks
def remove_blanks(cehd_data, **kwargs):
    '''
    Remove blanks from the 'BLANK_USED' variable 
    
    Other blanks identified later by 'QUALIFIER'.
    '''
    cehd_data = cehd_data.copy()
    not_blank = cehd_data['BLANK_USED'] == 'N'
    return cehd_data.loc[not_blank]
#endregion

# NOTE: This may not be needed
#region: initialize_elimination_log
def initialize_elimination_log(cehd_data):
    '''
    Initialize a dataframe to function as a log or tracker for the samples 
    eliminated during the data cleaning process.

    This log is named 'reasons' in the R script.
    '''
    return pd.DataFrame(
        index=pd.RangeIndex(min(cehd_data['YEAR']), max(cehd_data['YEAR'])+1)
    )
#endregion

# TODO: Double check whether these are all relevant
#region: pre_clean
def pre_clean(cehd_data, **kwargs):
    '''
    '''
    cehd_data = cehd_data.copy()
        
    cehd_data['AIR_VOLUME_SAMPLED'] = pd.to_numeric(
        cehd_data['AIR_VOLUME_SAMPLED'], errors='coerce'
        )

    cehd_data['BLANK_USED'] = factor(
        cehd_data['BLANK_USED'], categories=['Y', 'N']
        )

    cehd_data['CITY'] = as_character(cehd_data['CITY'])

    cehd_data['DATE_REPORTED'] = convert_date(cehd_data['DATE_REPORTED'])
    cehd_data['DATE_SAMPLED'] = convert_date(cehd_data['DATE_SAMPLED'])

    cehd_data['EIGHT_HOUR_TWA_CALC'] = factor(
        cehd_data['EIGHT_HOUR_TWA_CALC'], categories=['Y', 'N']
        )

    cehd_data['ESTABLISHMENT_NAME'] = as_character(
        cehd_data['ESTABLISHMENT_NAME']
        )
    cehd_data['FIELD_NUMBER'] = as_character(cehd_data['FIELD_NUMBER'])

    # NOTE: Seems unnecessary to go from one type to another
    cehd_data['IMIS_SUBSTANCE_CODE'] = factor(
        cehd_data['IMIS_SUBSTANCE_CODE'].str.replace(' ', '0').str.zfill(4)
    )
    cehd_data['IMIS_SUBSTANCE_CODE'] = as_character(
        cehd_data['IMIS_SUBSTANCE_CODE']
        )
    
    cehd_data['INSPECTION_NUMBER'] = factor(cehd_data['INSPECTION_NUMBER'])
    cehd_data['INSPECTION_NUMBER'] = as_character(cehd_data['INSPECTION_NUMBER'])

    cehd_data['INSTRUMENT_TYPE'] = as_character(cehd_data['INSTRUMENT_TYPE'])
    cehd_data['LAB_NUMBER'] = factor(cehd_data['LAB_NUMBER'])

    cehd_data['NAICS_CODE'] = (
        as_character(cehd_data['NAICS_CODE'])
        .apply(
            lambda x: x if isinstance(x, str) and len(x) >= 6 else np.nan)
    )

    cehd_data['OFFICE_ID'] = factor(cehd_data['OFFICE_ID'])
    cehd_data['QUALIFIER'] = as_character(cehd_data['QUALIFIER'])

    cehd_data['SAMPLE_RESULT'] = pd.to_numeric(
        cehd_data['SAMPLE_RESULT'], errors='coerce'
        )

    cehd_data['SAMPLE_TYPE'] = factor(cehd_data['SAMPLE_TYPE'])
    cehd_data['SAMPLE_WEIGHT'] = pd.to_numeric(
        cehd_data['SAMPLE_WEIGHT'], errors='coerce'
        )
    cehd_data['SAMPLING_NUMBER'] = factor(cehd_data['SAMPLING_NUMBER'])
    cehd_data['SAMPLING_NUMBER'] = as_character(cehd_data['SAMPLING_NUMBER'])

    cehd_data['SIC_CODE'] = factor(cehd_data['SIC_CODE'])
    cehd_data['STATE'] = factor(cehd_data['STATE'])
    cehd_data['SUBSTANCE'] = as_character(cehd_data['SUBSTANCE'])

    cehd_data['TIME_SAMPLED'] = pd.to_numeric(
        cehd_data['TIME_SAMPLED'], errors='coerce'
        )
    cehd_data['UNIT_OF_MEASUREMENT'] = as_character(
        cehd_data['UNIT_OF_MEASUREMENT']
        )

    cehd_data['ZIP_CODE'] = (
        as_character(cehd_data['ZIP_CODE'])
        .str.replace(' ', '0').str.zfill(5)
    )
    cehd_data['ZIP_CODE'] = factor(cehd_data['ZIP_CODE'])

    cehd_data['YEAR'] = factor(cehd_data['DATE_SAMPLED'].dt.year)

    cehd_data['INSPECTION_NUMBER'] = cehd_data['INSPECTION_NUMBER'].str.strip()
    cehd_data['SAMPLING_NUMBER'] = cehd_data['SAMPLING_NUMBER'].str.strip()

    return cehd_data
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