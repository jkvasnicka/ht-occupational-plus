'''
This module implements a data cleaning methodology by Jérôme Lavoué - 
Université de Montréal for the Chemical Exposure Health Data (CEHD).

The original R script was translated to Python.
'''
import pandas as pd
import numpy as np

# Constants
MILLIGRAMS_PER_GRAM = 1000
LITERS_PER_CUBIC_METER = 1000

#region: clean_cehd_data
def clean_cehd_data(database, path_settings):
    '''
    '''
    database = pre_clean(database)

    database = remove_blanks(database)
    database = remove_nonpersonal(database)
    database = exclude_few_samples(database)

    database = replace_missing_values(database, 'QUALIFIER')
    qualif_conv_2020 = load_qualifier_conversion(
        path_settings['qualif_conv_file']
        )

    database = replace_missing_values(database, 'UNIT_OF_MEASUREMENT')
    unit_conv_2020 = load_unit_measure_conversion(
        path_settings['unit_conv_file']
    )

    database = add_censored_column(database)

    database = remove_invalid_nd_samples(database, qualif_conv_2020)

    database = clean_unit_of_measurement(database, unit_conv_2020)

    database = remove_blk_not_bulk(database, qualif_conv_2020)

    database = remove_uninterpretable_qualifiers(database, qualif_conv_2020)

    database = remove_conflicting_qualifiers(database, qualif_conv_2020)

    database = remove_blk_possible_bulk_not_blank(database, qualif_conv_2020)

    database = remove_combustion_related_samples(database)

    database = remove_fibers_substance_conflict(database)

    database = remove_yttrium_substance_conflict(database)

    database = remove_approximate_measures(database)

    database = remove_qualifier_unit_mismatch(database)

    database = remove_invalid_unit_f_samples(database)

    database = remove_empty_unit_non_null_result(database)

    database = remove_percent_greater_than_100(database)

    database = create_detection_indicator(database)

    database = remove_invalid_units_for_all_substances(database)

    database = convert_percent_to_mass_concentration(database)

    database = remove_samples_with_missing_office_id(database)

    database = remove_samples_with_missing_time_sampled(database)

    database = remove_samples_with_null_time_sampled(database)

    database = remove_negative_sample_results(database)

    # database = remove_missing_sample_number(database)
    
    return database
#endregion

#region: remove_missing_sample_number
def remove_missing_sample_number(database):
    '''
    Remove samples that have a missing or null sampling number.
    '''
    database = database.copy()

    # NOTE: This deviates from the R script but may be needed to handle mixed 
    # dtypes like strings. 
    database['SAMPLING_NUMBER'] = pd.to_numeric(
        database['SAMPLING_NUMBER'], 
        errors='coerce'
        )
    rows_to_exclude = (
        database['SAMPLING_NUMBER'].isna() 
        | (database['SAMPLING_NUMBER'] == 0.)
    )

    return database[~rows_to_exclude]
#endregion

#region: remove_negative_sample_results
def remove_negative_sample_results(database):
    '''
    Remove samples with a sample result less than zero.
    '''
    database = database.copy()
    rows_to_exclude = database['SAMPLE_RESULT_3'] < 0.
    return database[~rows_to_exclude]
#endregion

#region: remove_samples_with_null_time_sampled
def remove_samples_with_null_time_sampled(database):
    '''
    Remove samples that have a null time sampled variable.
    '''
    database = database.copy()
    rows_to_exclude = database['TIME_SAMPLED'] == 0.
    return database[~rows_to_exclude]
#endregion

#region: remove_samples_with_missing_time_sampled
def remove_samples_with_missing_time_sampled(database):
    '''
    Remove samples that have a missing value for the time sampled variable.
    '''
    database = database.copy()
    rows_to_exclude = database['TIME_SAMPLED'].isna()
    return database[~rows_to_exclude]
#endregion

#region: remove_samples_with_missing_office_id
def remove_samples_with_missing_office_id(database):
    '''
    Remove samples that have a missing value for the office ID.
    '''
    database = database.copy()
    rows_to_exclude = database['OFFICE_ID'].isna()
    return database[~rows_to_exclude]
#endregion

#region: convert_percent_to_mass_concentration
def convert_percent_to_mass_concentration(database):
    '''
    Convert sample results from percentage concentration to mass concentration 
    (mg/m³).
    '''
    database = database.copy()

    database = remove_samples_with_null_weight(database)

    where_to_convert = (
        (database['SAMPLE_WEIGHT_2'] != 0) 
        & (database['UNIT_OF_MEASUREMENT_2'] == '%') 
        & (database['SAMPLE_RESULT_2'] > 0) 
        & database['SAMPLE_WEIGHT_2'].notna() 
        & database['AIR_VOLUME_SAMPLED'].notna() 
        & (database['AIR_VOLUME_SAMPLED'] > 0)
    )

    sample_result = database.loc[where_to_convert, 'SAMPLE_RESULT_2']
    sample_weight = database.loc[where_to_convert, 'SAMPLE_WEIGHT_2']
    air_volume_sampled = database.loc[where_to_convert, 'AIR_VOLUME_SAMPLED']

    conversion_factor = MILLIGRAMS_PER_GRAM * LITERS_PER_CUBIC_METER
    converted_result = (
        (sample_result * sample_weight * conversion_factor) 
        / air_volume_sampled
    )

    # Assign the converted results back to the dataframe
    database['SAMPLE_RESULT_3'] = database['SAMPLE_RESULT_2']
    database.loc[where_to_convert, 'SAMPLE_RESULT_3'] = converted_result

    database.loc[where_to_convert, 'UNIT_OF_MEASUREMENT_2'] = 'M.from.Perc'

    return database
#endregion

#region: remove_samples_with_null_weight
def remove_samples_with_null_weight(database):
    '''
    Remove samples where unit of measurement is percentage ('%'), the sample 
    result is non-null, but the sample weight is null.
    '''
    database = database.copy()

    # TODO: Is this step necessary?
    database['SAMPLE_WEIGHT_2'] = database['SAMPLE_WEIGHT'].fillna(0)

    rows_to_exclude = (
        (database['SAMPLE_WEIGHT_2'] == 0) &
        (database['UNIT_OF_MEASUREMENT_2'] == '%') &
        (database['SAMPLE_RESULT_2'] > 0)
    )

    return database.loc[~rows_to_exclude]
#endregion

# TODO: Remove hardcoding?
#region: remove_invalid_units_for_all_substances
def remove_invalid_units_for_all_substances(database):
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
    database = remove_invalid_units_for_substances(
        database, 
        top_substances, 
        valid_units_n31
        )

    valid_units_n32 = ['', '%', 'M']
    database = remove_invalid_units_for_substances(
        database, 
        ['9010'], 
        valid_units_n32
        )

    where_other_substances = (
        ~database['IMIS_SUBSTANCE_CODE'].isin(top_substances + ['9010'])
        )
    other_substances = list(
        database.loc[where_other_substances, 'IMIS_SUBSTANCE_CODE'].unique()
        )
    valid_units_n33 = ['', '%', 'M', 'P', 'F']
    database = remove_invalid_units_for_substances(
        database, 
        other_substances, 
        valid_units_n33
        )

    return database
#endregion

#region: remove_invalid_units_for_substances
def remove_invalid_units_for_substances(
        database, 
        substance_codes, 
        valid_units
        ):
    '''
    Remove samples where the unit of measurement is invalid for given
    substances.
    '''
    database = database.copy()

    where_in_substance_codes = (
        database['IMIS_SUBSTANCE_CODE'].isin(substance_codes)
    )
    where_invalid_units = ~database['UNIT_OF_MEASUREMENT_2'].isin(valid_units)

    rows_to_exclude = where_in_substance_codes & where_invalid_units
    return database.loc[~rows_to_exclude]
#endregion

#region: create_detection_indicator
def create_detection_indicator(database):
    '''
    Create a new column 'QUALIFIER_2' to indicate detection status.
    '''
    database = database.copy()

    database['QUALIFIER_2'] = 'detected'  # initialize
    database.loc[database['SAMPLE_RESULT_2'] == 0, 'QUALIFIER_2'] = 'ND'

    return database
#endregion

#region: remove_percent_greater_than_100
def remove_percent_greater_than_100(database):
    '''
    Remove samples where the unit of measurement is '%' and the sample result
    is greater than 100.
    '''
    rows_to_exclude = (
        (database['UNIT_OF_MEASUREMENT_2'] == '%') &
        (database['SAMPLE_RESULT_2'] > 100.)
    )
    
    return database.loc[~rows_to_exclude]
#endregion

#region: remove_empty_unit_non_null_result
def remove_empty_unit_non_null_result(database):
    '''
    Remove samples where the unit of measurement is empty and the sample 
    result is not null.
    '''
    rows_to_exclude = (
        (database['UNIT_OF_MEASUREMENT_2'] == '') &
        (database['SAMPLE_RESULT_2'] > 0)
    )
    
    return database.loc[~rows_to_exclude]
#endregion

#region: remove_invalid_unit_f_samples
def remove_invalid_unit_f_samples(database):
    '''
    Remove samples with specific substance codes that should not have "F" as
    the unit of measurement.
    '''
    # These codes should not have "F" as the unit of measurement
    invalid_substance_codes = ['1073', '2270', '2470', '9135']
    
    where_invalid_units = (
        (database['UNIT_OF_MEASUREMENT_2'] == 'F') &
        (database['IMIS_SUBSTANCE_CODE'].isin(invalid_substance_codes))
    )
    
    return database.loc[~where_invalid_units]
#endregion

#region: remove_qualifier_unit_mismatch
def remove_qualifier_unit_mismatch(database):
    '''
    Remove samples with inconsistent qualifier and unit of measurement.
    '''
    database = database.copy()

    condition_inconsistent_units = (
        (database['UNIT_OF_MEASUREMENT_2'] != '%') 
        & (database['QUALIFIER'] == '%')
    ) | (
        (database['UNIT_OF_MEASUREMENT_2'] != 'M') 
        & (database['QUALIFIER'] == 'M')
    )

    return database.loc[~condition_inconsistent_units]
#endregion

#region: remove_approximate_measures
def remove_approximate_measures(database):
    '''
    Remove samples where the QUALIFIER indicates an approximate measure.
    '''
    database = database.copy()
    
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
    rows_to_exclude = database['QUALIFIER'].isin(approximate_qualifiers)
    
    return database.loc[~rows_to_exclude]
#endregion

#region: remove_yttrium_substance_conflict
def remove_yttrium_substance_conflict(database):
    '''
    Remove samples where the qualifier 'Y' is used but the substance code is
    not 9135.
    '''
    database = database.copy()

    rows_to_exclude = (
        (database['QUALIFIER'] == 'Y') 
        & (database['IMIS_SUBSTANCE_CODE'] != '9135')
    )

    return database.loc[~rows_to_exclude]
#endregion

#region: remove_fibers_substance_conflict
def remove_fibers_substance_conflict(database):
    '''
    Remove samples where the qualifier suggests fibers (F) but the substance
    code is not 9020.
    '''
    database = database.copy()
    rows_to_exclude = (
        (database['QUALIFIER'] == 'F') 
        & (database['IMIS_SUBSTANCE_CODE'] != '9020')
    )
    return database.loc[~rows_to_exclude]
#endregion

#region: remove_combustion_related_samples
def remove_combustion_related_samples(database):
    '''
    Remove samples with qualifiers related to combustion.
    '''
    database = database.copy()

    combustion_qualifiers = ['COMB', 'COMD', 'com', 'comb']
    rows_to_exclude = database['QUALIFIER'].isin(combustion_qualifiers)

    return database.loc[~rows_to_exclude]
#endregion

#region: remove_blk_possible_bulk_not_blank
def remove_blk_possible_bulk_not_blank(database, qualif_conv_2020):
    '''
    Remove samples judged to be possible blank (BLK) and bulk, yet BLANK_USED
    is 'N'.
    '''
    database = database.copy()

    condition_blk_possible_bulk = (
        (qualif_conv_2020['clean'] == 'BLK')
        & (qualif_conv_2020['possible_bulk'] == 'Y')
    )
    rows_to_exclude = rows_to_exclude_based_on_qualifier(
        database, 
        qualif_conv_2020, 
        condition_blk_possible_bulk
        )

    # Further filter rows where BLANK_USED is 'N'
    rows_to_exclude = rows_to_exclude & (database['BLANK_USED'] == 'N')

    return database.loc[~rows_to_exclude]
#endregion

#region: remove_conflicting_qualifiers
def remove_conflicting_qualifiers(database, qualif_conv_2020):
    '''
    Remove samples with qualifiers conflicting with sample type.
    '''
    database = database.copy()

    where_conflict = qualif_conv_2020['clean'].isin(['B', 'W'])
    database = remove_samples_based_on_qualifier(
        database, 
        qualif_conv_2020, 
        where_conflict
        )
    return database
#endregion

#region: remove_uninterpretable_qualifiers
def remove_uninterpretable_qualifiers(database, qualif_conv_2020):
    '''
    Remove samples with qualifiers deemed uninterpretable.
    '''
    database = database.copy()

    where_eliminate = qualif_conv_2020['clean'] == 'eliminate'
    database = remove_samples_based_on_qualifier(
        database, 
        qualif_conv_2020, 
        where_eliminate
        )
    return database
#endregion

#region: remove_blk_not_bulk
def remove_blk_not_bulk(database, qualif_conv_2020):
    '''
    Remove samples where QUALIFIER is 'BLK' and not possible bulk.
    '''
    database = database.copy()

    where_blk_not_bulk  = (
        (qualif_conv_2020['clean'] == 'BLK') 
        & (qualif_conv_2020['possible_bulk'] == 'N')
    )
    database = remove_samples_based_on_qualifier(
        database, 
        qualif_conv_2020, 
        where_blk_not_bulk
        )
    return database
#endregion

#region: remove_samples_based_on_qualifier
def remove_samples_based_on_qualifier(database, qualif_conv_2020, condition):
    '''
    General function to remove samples based on QUALIFIER conditions.
    '''
    database = database.copy()

    rows_to_exclude = rows_to_exclude_based_on_qualifier(
        database, 
        qualif_conv_2020, 
        condition
        )
    return database.loc[~rows_to_exclude]
#endregion:

#region: rows_to_exclude_based_on_qualifier
def rows_to_exclude_based_on_qualifier(database, qualif_conv_2020, condition):
    '''
    General function to remove samples based on QUALIFIER conditions.
    '''
    database = database.copy()
    
    raw_values_to_exclude = qualif_conv_2020.loc[condition, 'raw']
    return database['QUALIFIER'].isin(raw_values_to_exclude)
#endregion

#region: clean_unit_of_measurement
def clean_unit_of_measurement(database, unit_conv_2020):
    '''
    Clean the `UNIT_OF_MEASUREMENT` column by mapping raw values to clean 
    values.
    '''
    database = database.copy()

    # Initialize a cleaned column
    database['UNIT_OF_MEASUREMENT_2'] = database['UNIT_OF_MEASUREMENT']
    
    for clean_value in unit_conv_2020['clean'].unique():
        where_clean_value = unit_conv_2020['clean'] == clean_value
        raw_values = list(unit_conv_2020.loc[where_clean_value, 'raw'])
        where_needs_clean = database['UNIT_OF_MEASUREMENT'].isin(raw_values)
        database.loc[where_needs_clean, 'UNIT_OF_MEASUREMENT_2'] = clean_value

    return database
#endregion

#region: remove_invalid_nd_samples
def remove_invalid_nd_samples(database, qualif_conv_2020):
    '''
    Remove samples where `QUALIFIER` suggests ND but `SAMPLE_RESULT_2` > 0
    and not censored (N08), and where `QUALIFIER` suggests ND or is censored
    but `SAMPLE_RESULT_2` > 0 (N29).
    '''
    database = database.copy()

    where_nd = qualif_conv_2020['clean'] == 'ND'
    nd_qualifiers = qualif_conv_2020.loc[where_nd, 'raw']
    condition_n08 = (
        (database['SAMPLE_RESULT_2'] > 0) 
        & (database['CENSORED'] != 'Y') 
        & (database['QUALIFIER'].isin(nd_qualifiers))
    )
    database = database[~condition_n08]  # N08

    condition_n29 = (
        (database['SAMPLE_RESULT_2'] > 0) 
        & ((database['CENSORED'] == 'Y') 
           | (database['QUALIFIER'].isin(nd_qualifiers)))
    )
    
    database = database[~condition_n29]  # N29

    return database
#endregion

#region: add_censored_column
def add_censored_column(database):
    '''
    Add a column indicating that the sample is censored ONLY based on the 
    'QUALIFIER' column.
    '''
    database = database.copy()

    new_column = 'CENSORED'
    database[new_column] = 'N'  # initialize
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
    where_censored = database['QUALIFIER'].isin(qualifier_censored_values)
    database.loc[where_censored, new_column] = 'Y'

    # FIXME: Something seems odd. Replacing NA with 'raw was NA' and then ''
    database['QUALIFIER'] = database['QUALIFIER'].replace('raw was NA', '')
    database['SAMPLE_RESULT_2'] = database['SAMPLE_RESULT'].fillna(0)

    return database
#endregion

#region: replace_missing_values
def replace_missing_values(database, column):
    '''
    '''
    database = database.copy()
    database[column] = database[column].fillna('raw was NA')
    return database
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

#region: exclude_few_samples
def exclude_few_samples(database):
    '''
    Exclude substances with few samples or non-chemical IMIS codes.
    '''
    database = database.copy()

    ## Exclude substances with few samples
    subst = database['IMIS_SUBSTANCE_CODE'].value_counts().reset_index()
    subst.columns = ['code', 'n']
    where_enough_samples = subst['n'] >= 100
    subst = subst[where_enough_samples]

    ## Remove non-chemical substance codes
    # FIXME: Remove hardcoding
    non_chemical_codes = [
        'G301', 'G302', 'Q115', 'T110', 'M125', 'Q116', 'Q100', 'S325'
        ]
    where_non_chemical = subst['code'].isin(non_chemical_codes)
    subst = subst[~where_non_chemical]

    sub_list_all = list(subst['code'])
    return database[database['IMIS_SUBSTANCE_CODE'].isin(sub_list_all)]
#endregion

#region: remove_nonpersonal
def remove_nonpersonal(database):
    '''
    Exclude all samples that are not designated as 'P'.
    '''
    database = database.copy()
    not_blank = database['SAMPLE_TYPE'] != 'P'
    return database.loc[~not_blank]
#endregion

#region: remove_blanks
def remove_blanks(database):
    '''
    Remove blanks from the 'BLANK_USED' variable 
    
    Other blanks identified later by 'QUALIFIER'.
    '''
    database = database.copy()
    not_blank = database['BLANK_USED'] == 'N'
    return database.loc[not_blank]
#endregion

# NOTE: This may not be needed
#region: initialize_elimination_log
def initialize_elimination_log(database):
    '''
    Initialize a dataframe to function as a log or tracker for the samples 
    eliminated during the data cleaning process.

    This log is named 'reasons' in the R script.
    '''
    return pd.DataFrame(
        index=pd.RangeIndex(min(database['YEAR']), max(database['YEAR'])+1)
    )
#endregion

# TODO: Double check whether these are all relevant
#region: pre_clean
def pre_clean(database):
    '''
    '''
    database = database.copy()
        
    database['AIR_VOLUME_SAMPLED'] = pd.to_numeric(
        database['AIR_VOLUME_SAMPLED'], errors='coerce'
        )

    database['BLANK_USED'] = factor(
        database['BLANK_USED'], categories=['Y', 'N']
        )

    database['CITY'] = as_character(database['CITY'])

    database['DATE_REPORTED'] = convert_date(database['DATE_REPORTED'])
    database['DATE_SAMPLED'] = convert_date(database['DATE_SAMPLED'])

    database['EIGHT_HOUR_TWA_CALC'] = factor(
        database['EIGHT_HOUR_TWA_CALC'], categories=['Y', 'N']
        )

    database['ESTABLISHMENT_NAME'] = as_character(
        database['ESTABLISHMENT_NAME']
        )
    database['FIELD_NUMBER'] = as_character(database['FIELD_NUMBER'])

    # NOTE: Seems unnecessary to go from one type to another
    database['IMIS_SUBSTANCE_CODE'] = factor(
        database['IMIS_SUBSTANCE_CODE'].str.replace(' ', '0').str.zfill(4)
    )
    database['IMIS_SUBSTANCE_CODE'] = as_character(
        database['IMIS_SUBSTANCE_CODE']
        )
    
    database['INSPECTION_NUMBER'] = factor(database['INSPECTION_NUMBER'])
    database['INSPECTION_NUMBER'] = as_character(database['INSPECTION_NUMBER'])

    database['INSTRUMENT_TYPE'] = as_character(database['INSTRUMENT_TYPE'])
    database['LAB_NUMBER'] = factor(database['LAB_NUMBER'])

    database['NAICS_CODE'] = (
        as_character(database['NAICS_CODE'])
        .apply(
            lambda x: x if isinstance(x, str) and len(x) >= 6 else np.nan)
    )

    database['OFFICE_ID'] = factor(database['OFFICE_ID'])
    database['QUALIFIER'] = as_character(database['QUALIFIER'])

    database['SAMPLE_RESULT'] = pd.to_numeric(
        database['SAMPLE_RESULT'], errors='coerce'
        )

    database['SAMPLE_TYPE'] = factor(database['SAMPLE_TYPE'])
    database['SAMPLE_WEIGHT'] = pd.to_numeric(
        database['SAMPLE_WEIGHT'], errors='coerce'
        )
    database['SAMPLING_NUMBER'] = factor(database['SAMPLING_NUMBER'])
    database['SAMPLING_NUMBER'] = as_character(database['SAMPLING_NUMBER'])

    database['SIC_CODE'] = factor(database['SIC_CODE'])
    database['STATE'] = factor(database['STATE'])
    database['SUBSTANCE'] = as_character(database['SUBSTANCE'])

    database['TIME_SAMPLED'] = pd.to_numeric(
        database['TIME_SAMPLED'], errors='coerce'
        )
    database['UNIT_OF_MEASUREMENT'] = as_character(
        database['UNIT_OF_MEASUREMENT']
        )

    database['ZIP_CODE'] = (
        as_character(database['ZIP_CODE'])
        .str.replace(' ', '0').str.zfill(5)
    )
    database['ZIP_CODE'] = factor(database['ZIP_CODE'])

    database['YEAR'] = factor(database['DATE_SAMPLED'].dt.year)

    database = trim_white_spaces(database)

    return database
#endregion

#region: trim_white_spaces
def trim_white_spaces(database):
    '''
    '''
    database['INSPECTION_NUMBER'] = database['INSPECTION_NUMBER'].str.strip()
    database['SAMPLING_NUMBER'] = database['INSPECTION_NUMBER'].str.strip()
    return database
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
    diff_columns = []
    for column in df1.columns:
        if column in df2.columns:
            if not df1[column].equals(df2[column]):
                diff_columns.append(column)
        else:
            print(f"Column {column} is not present in both dataframes.")
    return diff_columns
#endregion