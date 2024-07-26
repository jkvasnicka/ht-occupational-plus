'''
This module implements a data cleaning methodology by Jérôme Lavoué - 
Université de Montréal for the Chemical Exposure Health Data (CEHD).

The original R script was translated to Python.
'''
import pandas as pd
import numpy as np

#region: clean_cehd_data
def clean_cehd_data(database, path_settings):
    '''
    '''
    database = pre_clean(database)

    database = remove_blanks(database)
    database = remove_nonpersonal(database)
    database = exclude_few_samples(database)

    replace_missing_values(database, 'QUALIFIER')
    qualif_conv_2020 = load_qualifier_conversion(
        path_settings['qualif_conv_file']
        )

    replace_missing_values(database, 'UNIT_OF_MEASUREMENT')
    unit_conv_2020 = load_unit_measure_conversion(
        path_settings['unit_conv_file']
    )

    return database
#endregion

#region: add_censored_column
def add_censored_column(database):
    '''
    Add a column indicating that the sample is censored ONLY based on the 
    'QUALIFIER' column.
    '''
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
#endregion

#region: replace_missing_values
def replace_missing_values(database, column):
    '''
    '''
    database[column] = database[column].fillna('raw was NA')
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
    Initialize a dataframe to function as a log or tracker for the records 
    eliminated during the data cleaning process.

    This log is named 'reasons' in the R script.
    '''
    return pd.DataFrame(
        index=pd.RangeIndex(min(database['YEAR']), max(database['YEAR'])+1)
    )
#endregion

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