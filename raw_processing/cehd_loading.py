'''
This module contains functions for loading the Chemical Exposure Health Data 
(CEHD). This is a minimalistic module that simply loads the data without 
applying any transformations (beyond I/O necessities like concatenating 
files). The logic for cleaning and preprocessing the CEHD can be found in a
separate module.
'''

import os
import chardet
import pandas as pd
import numpy as np

from . import osha_loading

#region: raw_chem_exposure_health_data
def raw_chem_exposure_health_data(cehd_settings, path_settings):
    '''
    Load the raw Chemical Exposure Health Data (CEHD).

    This function serves as the top-level interface for loading the raw CEHD.
    The function first checks if the yearly CEHD releases have already been 
    combined into a single DataFrame. If raw_cehd_file is absent but 
    raw_cehd_dir is provided, the loader combines all yearly CEHD files 
    and writes a Feather cache for future runs. Provide at least one of these 
    paths; otherwise the workflow will raise an error.

    Returns
    -------
    pandas.DataFrame
    '''
    file_path = path_settings.get('raw_cehd_file')
    dir_path  = path_settings.get('raw_cehd_dir')

    if file_path and os.path.isfile(file_path):
        # Load the pre-prepared CEHD data (combined across years)
        return _raw_cehd_from_single_file(file_path)
        
    if dir_path and os.path.isdir(dir_path):
        print(
            '    Preparing Chemical Exposure Health Data from multiple files.'
            )
        exposure_data = _raw_cehd_from_multiple_files(
            dir_path, 
            cehd_settings['file_name_prefix'],
            cehd_settings['rename_mapper'],
            cehd_settings['initial_dtypes']
            )
        if file_path:  # cache for next time
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Preserves non-default index
            exposure_data.reset_index().to_feather(file_path)
        return exposure_data

    raise FileNotFoundError(
        'Cannot locate raw CEHD files. Neither "raw_cehd_file" nor',
        '"raw_cehd_dir" exists.'
        )
#endregion

#region _raw_cehd_from_multiple_files
def _raw_cehd_from_multiple_files(
        raw_cehd_dir, 
        file_name_prefix,
        rename_mapper,
        initial_dtypes
        ):
    '''
    Load the Chemical Exposure Health Data (CEHD) into a single DataFrame.

    This function walks through the specified directory, identifies files
    following the naming convention 'sample_data_YEAR.extension', where
    'YEAR' is a four-digit year and 'extension' is either 'csv', 'txt', or 
    'xml'.

    Parameters
    ----------
    raw_cehd_dir : str
        The directory containing the OSHA files.
    rename_mapper : dict
        The dictionary used to rename columns.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame.
    '''
    exposure_data = []  # initialize
    for root, _, files in os.walk(raw_cehd_dir):
        for file in files:
            if file.startswith(file_name_prefix):
                parts = file.split('.')
                extension = parts[-1]
                if extension in ['csv', 'txt', 'xml']:
                    # The file contains OSHA data
                    year = parts[0].split('_')[-1]
                    print(f'      Loading {year} data...')
                    if extension in ['csv', 'txt']:
                        year_data = _cehd_from_csv(
                            root, file, rename_mapper
                            )
                    elif extension == 'xml':
                        year_data = _cehd_from_xml(
                            root, file, rename_mapper
                            )
                    # Create a new column with the file year
                    year_data['YEAR'] = year
                    exposure_data.append(year_data)
    exposure_data = pd.concat(exposure_data, ignore_index=True)
    exposure_data = pre_clean(exposure_data, initial_dtypes)
    return exposure_data
#endregion

#region: _raw_cehd_from_single_file
def _raw_cehd_from_single_file(raw_cehd_file):
    return pd.read_feather(raw_cehd_file).set_index('index')
#endregion

#region: _cehd_from_csv
def _cehd_from_csv(root, file, rename_mapper):
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

    Notes
    -----
    on_bad_lines='skip' was introduced to handle a few bad lines in the 2024
    release of the CEHD. These lines had extra '|' characters in the raw file.
    '''
    raw_cehd_file = os.path.join(root, file)

    # Determine encoding dynamically
    with open(raw_cehd_file, 'rb') as file:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']
    delimiter = _determine_delimiter(raw_cehd_file, encoding)
    
    csv_data = pd.read_csv(
        raw_cehd_file, 
        encoding=encoding, 
        delimiter=delimiter,
        low_memory=False,
        on_bad_lines='skip'
    )

    csv_data = _standardize(csv_data, rename_mapper)

    return csv_data
#endregion

#region: _determine_delimiter
def _determine_delimiter(file_path, encoding):
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

#region: _cehd_from_xml
def _cehd_from_xml(root, file, rename_mapper):
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
    raw_cehd_file = os.path.join(root, file)
    
    xml_data = pd.read_xml(raw_cehd_file)

    xml_data = _standardize(xml_data, rename_mapper)
    
    return xml_data
#endregion

#region: _standardize
def _standardize(exposure_data, rename_mapper):
    '''
    Standardize the DataFrame for concatenation.

    Parameters
    ----------
    exposure_data : pd.DataFrame
        The DataFrame to clean.
    rename_mapper : dict
        The dictionary used to rename columns.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    '''
    exposure_data.columns = exposure_data.columns.str.upper()
    exposure_data = exposure_data.rename(rename_mapper, axis=1)
    exposure_data = _strip_trailing_commas(exposure_data)
    return exposure_data
#endregion

#region: _strip_trailing_commas
def _strip_trailing_commas(df):
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

#region: pre_clean
def pre_clean(exposure_data, initial_dtypes):
    '''
    Apply minimal data cleaning and type conversion to ensure that the data is 
    in a workable format.

    Minimal pre-cleaning is done to address issues related to mixed data types
    within columns, which can cause problems when writing to formats like 
    Feather. Pandas requires consistent data types.
    '''
    exposure_data = exposure_data.copy()

    exposure_data = osha_loading.pre_clean(exposure_data, initial_dtypes)

    exposure_data = exposure_data.sort_index(axis=1)

    exposure_data['IMIS_SUBSTANCE_CODE'] = (
        exposure_data['IMIS_SUBSTANCE_CODE'].str.replace(' ', '0').str.zfill(4)
    )

    exposure_data['NAICS_CODE'] = (
        exposure_data['NAICS_CODE'].apply(
            lambda x: x if isinstance(x, str) and len(x) >= 6 else np.nan
            )
        )

    exposure_data['ZIP_CODE'] = (
        exposure_data['ZIP_CODE']
        .str.replace(' ', '0').str.zfill(5)
    )

    exposure_data['YEAR'] = (
        _replace_file_year_with_sampled_year(
            exposure_data['YEAR'],
            exposure_data['DATE_SAMPLED'])
    )

    exposure_data['INSPECTION_NUMBER'] = (
        exposure_data['INSPECTION_NUMBER'].str.strip()
    )

    exposure_data['SAMPLING_NUMBER'] = (
        exposure_data['SAMPLING_NUMBER'].str.strip()
    )

    return exposure_data
#endregion

#region: _replace_file_year_with_sampled_year
def _replace_file_year_with_sampled_year(file_year, date_sampled):
    '''
    Update the 'YEAR' column according to the 'DATE_SAMPLED'. 

    Assumes the 'YEAR' column was prefilled with the file year based on the
    filename (e.g., 'sample_data_[file_year].csv'). Replaces the file years 
    with the year sampled, where values in 'YEAR_SAMPLED' are not missing.
    '''
    year_sampled = date_sampled.dt.year
    return file_year.where(year_sampled.isna(), year_sampled).astype('int64')
#endregion