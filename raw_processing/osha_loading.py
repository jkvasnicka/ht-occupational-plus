'''
Common functions for loading OSHA datasets.
'''

import pandas as pd
from copy import deepcopy

#region: pre_clean
def pre_clean(exposure_data, initial_dtypes):
    '''
    Apply minimal data cleaning and type conversion to ensure that the data is 
    in a workable format.

    Notes
    -----
    This function is intended to be augmented with additional pre-cleaning for 
    specific datasets, e.g., CEHD and USIS. 
    '''
    exposure_data = exposure_data.copy()

    exposure_data = set_initial_dtypes(exposure_data, initial_dtypes)

    return exposure_data
#endregion

#region: set_initial_dtypes
def set_initial_dtypes(exposure_data, initial_dtypes):
    '''
    Set consistent data types for each column based on the configuration 
    settings.

    Note
    ----
    "Initial" dtypes can be readily written and read from a file and 
    therefore do not include complex dtypes like pandas.Categorical, which are
    set post-loading.
    '''
    exposure_data = exposure_data.copy()
    initial_dtypes = deepcopy(initial_dtypes)
    
    for col, settings in initial_dtypes.items():
        
        dtype = settings.pop('dtype')

        if dtype == 'string':
            exposure_data[col] = to_string(exposure_data[col])
        elif dtype == 'datetime':
            exposure_data[col] = to_datetime(exposure_data[col])
        elif dtype == 'numeric':
            exposure_data[col] = pd.to_numeric(exposure_data[col], **settings)
        elif dtype == 'integer_string':
            exposure_data[col] = convert_to_integer_string(exposure_data[col])
        else:
            # Infer pandas dtype
            exposure_data[col] = exposure_data[col].astype(dtype)

    return exposure_data
#endregion

#region: to_string
def to_string(series):
    '''
    Convert a pandas Series to strings, while leaving NaNs unchanged.

    Note
    ----
    This function is used while pandas APIs for StringDtype and pd.NA are
    labeled as "experimental"
    '''
    return series.apply(lambda x: x if pd.isna(x) else str(x))
#endregion

#region: to_datetime
def to_datetime(series):
    '''
    Lowercase and date conversion handling multiple formats
    '''
    return pd.to_datetime(
        series.str.lower(),
        errors='coerce',
        format='%Y-%b-%d'
    ).combine_first(
        pd.to_datetime(series.str.lower(), errors='coerce', format='%Y/%m/%d')
    ).combine_first(
        pd.to_datetime(series.str.lower(), errors='coerce', format='%Y-%m-%d')
    .combine_first(
        pd.to_datetime(series.str.lower(), errors='coerce', format='%d-%b-%Y')
    )
    )
#endregion

#region: convert_to_integer_string
def convert_to_integer_string(series):
    '''
    Convert a pandas Series to integer strings where possible.
    NaNs and non-convertible strings are left unchanged.
    '''
    # Attempt to convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    integer_strings = numeric_series.dropna().astype('int').astype('str')
    
    # Where successfully converted, use the integer string
    series = series.where(numeric_series.isna(), integer_strings)
    
    return series
#endregion