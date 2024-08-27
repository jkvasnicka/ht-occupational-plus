'''
This module contains functions for loading and preprocessing Chemical Exposure
Health Data (CEHD) from OSHA. Logic for cleaning and preprocessing these data
can be found in a separate module.
'''

import os
import chardet
import pandas as pd

#region: load_osha_data_temp
def load_osha_data_temp(osha_file, do_overwrite=True):
    '''
    Temporary function for quickly loading and cleaning a pre-assembled
    OSHA dataframe.

    This bypasses loading individual dataframes for each year.
    '''
    osha_data = pd.read_csv(osha_file)
    osha_data = final_cleaning(osha_data)

    if do_overwrite:
        osha_data.to_csv(osha_file)

    return osha_data
#endregion

#region load_osha_data
def load_osha_data(osha_dir, rename_mapper):
    '''
    Loads the OSHA data into a single DataFrame.

    This function walks through the specified directory, identifies files
    following the naming convention 'sample_data_YEAR.extension', where
    'YEAR' is a four-digit year and 'extension' is either 'csv' or 'xml'.

    Parameters
    ----------
    osha_dir : str
        The directory containing the OSHA files.
    rename_mapper : dict
        The dictionary used to rename columns.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame.
    '''
    osha_data = []  # initialize
    for root, _, files in os.walk(osha_dir):
        for file in files:

            parts = file.split('.')
            extension = parts[-1]
            if extension in ['csv', 'xml']:
                # The file contains OSHA data
                year = parts[0].split('_')[-1]
                print(f'Loading {year} data...')
                if extension == 'csv':
                    year_data = load_csv_data(
                        root, file, rename_mapper
                        )
                elif extension == 'xml':
                    year_data = load_xml_data(
                        root, file, rename_mapper
                        )
                year_data['YEAR'] = year
                osha_data.append(year_data)
    osha_data = pd.concat(osha_data, ignore_index=True)
    osha_data = final_cleaning(osha_data)

    return osha_data
#endregion

#region: load_csv_data
def load_csv_data(root, file, rename_mapper):
    '''
    Loads CSV data for a given range of years.

    Parameters
    ----------
    year_range : range
        The range of years to load data for.

    Returns
    -------
    dict
        A dictionary of DataFrames for each year.
    '''
    osha_file = os.path.join(root, file)

    # Determine encoding dynamically
    with open(osha_file, 'rb') as file:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']
    delimiter = determine_delimiter(osha_file, encoding)
    
    csv_data = pd.read_csv(
        osha_file, 
        encoding=encoding, 
        delimiter=delimiter,
        low_memory=False
    )

    csv_data = clean_columns(csv_data, rename_mapper)

    return csv_data
#endregion

#region: determine_delimiter
def determine_delimiter(file_path, encoding):
    '''
    Determines the delimiter used in a file.

    Parameters
    ----------
    file_path : str
        The path to the file.
    encoding : str
        The encoding of the file.

    Returns
    -------
    str
        The delimiter used in the file.
    '''
    with open(file_path, 'r', encoding=encoding) as file:
        first_line = file.readline()
        if '|' in first_line:
            return '|'
        return ','
#endregion

#region: load_xml_data
def load_xml_data(root, file, rename_mapper):
    '''
    Loads XML data from a file.

    Parameters
    ----------
    root : str
        The root directory where the file is located.
    file : str
        The name of the file.
    rename_mapper : dict
        The dictionary used to rename columns.

    Returns
    -------
    dict
        A dictionary of DataFrames for each year.
    '''
    osha_file = os.path.join(root, file)
    
    xml_data = pd.read_xml(osha_file)

    xml_data = clean_columns(xml_data, rename_mapper)
    
    return xml_data
#endregion

#region: clean_columns
def clean_columns(osha_data, rename_mapper):
    '''
    Cleans column names and contents of the DataFrame.

    Parameters
    ----------
    osha_data : pd.DataFrame
        The DataFrame to clean.
    rename_mapper : dict
        The dictionary used to rename columns.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    '''
    osha_data.columns = osha_data.columns.str.upper()
    osha_data = osha_data.rename(rename_mapper, axis=1)
    osha_data = strip_trailing_commas(osha_data)
    return osha_data
#endregion

#region: strip_trailing_commas
def strip_trailing_commas(df):
    '''
    Identifies and cleans trailing commas from DataFrame columns and contents.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    '''
    columns_with_trailing_commas = {
        col : col.strip(',') for col in df
        if col.endswith(',')
    }

    for col in columns_with_trailing_commas:
        df[col] = df[col].str.strip(',')
    df = df.rename(columns_with_trailing_commas, axis=1)
    
    return df
#endregion

#region: final_cleaning
def final_cleaning(osha_data):
    '''
    Perform final cleaning steps.

    Parameters
    ----------
    osha_data : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.

    Notes
    -----
    This code was transcribed from an R script by Delphine Bosson-Rieutort, 
    based on scripts from Jérôme Lavoué.
    '''
    # NOTE: Some of this may be unnecessary.
    osha_data = osha_data.astype(str)  # 'character' in R
    osha_data = osha_data.reindex(sorted(osha_data.columns), axis=1)
    return osha_data
#endregion