'''
This module implements a data cleaning methodology by Jérôme Lavoué - 
Université de Montréal for the Chemical Exposure Health Data (CEHD).

The original R script was translated to Python.
'''
import pandas as pd
import numpy as np

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
        .apply(lambda x: x if len(x) >= 6 else np.nan)
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
    return column.astype(str) # .str.replace('"', '""').str.strip()
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