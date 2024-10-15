'''
'''

import pandas as pd
import numpy as np
import json

#region: OshaDataCleaner.__init__
class OshaDataCleaner:
    '''
    A base class for cleaning OSHA-related exposure datasets.

    This class defines common cleaning operations for OSHA datasets and 
    provides a framework for running a sequence of cleaning steps dynamically.
    '''
    def __init__(self, data_settings):
        self._data_settings = data_settings
#endregion

    # TODO: Remove short-duration samples?
    #region: clean_raw_data
    def clean_raw_data(
            self, 
            raw_exposure_data,
            cleaning_steps, 
            log_file=None, 
            do_log_changes=True
            ):
        '''
        Clean the raw exposure data using a sequence of cleaning steps.

        Parameters
        ----------
        raw_exposure_data : pandas.DataFrame
            The raw exposure data to be cleaned.
        cleaning_steps : list of str
            List of method names (as strings) to apply to the data.
        log_file : str or None, optional
            Path to a log file where changes to the data (e.g., rows removed) 
            will be saved. If None, no log file is created.
        do_log_changes : bool, optional
            If True, logs the changes made during the cleaning process to the 
            specified log file.

        Returns
        -------
        pandas.DataFrame
            The cleaned exposure data.
        '''
        exposure_data = raw_exposure_data.copy()
        change_log = {}  # initialize

        exposure_data = self.set_categorical_dtypes(
            exposure_data, 
            self._data_settings.get('categoricals', {})
            )
        
        for step_name in cleaning_steps:
            N_before = len(exposure_data)
            # Dynamically get the cleaning method from the step name
            exposure_data = getattr(self, step_name)(exposure_data)
            N_after = len(exposure_data)
            change_log[step_name] =  N_after - N_before

        if do_log_changes is True:
            with open(log_file, 'w') as log_file:
                json.dump(change_log, log_file, indent=4)

        return exposure_data
    #endregion

    #region: clean_duplicates
    def clean_duplicates(self, exposure_data):
        '''
        Clean the dataset by identifying and removing duplicate samples.

        1. Identifies and removes conflicting duplicates—records with the same 
        unique sample columns (i.e., they appear to describe the same sample) 
        but differing in comparison columns.
        2. Identifies true duplicates—identical records across all relevant columns.
        3. Retains only the first occurrence of true duplicates for substance '9010',
        while removing other exact duplicates.
        '''
        exposure_data = exposure_data.copy()

        unique_sample_columns = self._data_settings['unique_sample_columns']
        comparison_columns = self._data_settings['comparison_columns']
        substance_column = self._data_settings['substance_column']

        ## Step 1: Identify and remove conflicting duplicates

        where_unique_sample_duplicate = (
            exposure_data.duplicated(
                subset=unique_sample_columns, 
                keep=False
                )
        )
        conflicting_samples = (
            exposure_data.loc[where_unique_sample_duplicate]
            .drop_duplicates(
                subset=(
                    unique_sample_columns 
                    + comparison_columns
                ), 
                keep=False
                )
        )
        non_conflicting_data = exposure_data.drop(conflicting_samples.index)
        
        # Step 2: Handle true duplicates (exact matches) selectively

        where_true_duplicate = (
            where_unique_sample_duplicate.loc[non_conflicting_data.index]
        )
        
        duplicates_data = non_conflicting_data.loc[where_true_duplicate]
        non_duplicates_data = non_conflicting_data.loc[~where_true_duplicate]
        
        # For duplicates with substance '9010', keep only the first occurrence
        where_9010 = duplicates_data[substance_column] == '9010'
        duplicates_9010 = duplicates_data.loc[where_9010]
        duplicates_9010_deduped = (
            duplicates_9010.drop_duplicates(
                subset=unique_sample_columns, 
                keep='first'
                )
        )
            
        return pd.concat([non_duplicates_data, duplicates_9010_deduped])
    #endregion

    #region: remove_nonpersonal
    def remove_nonpersonal(self, exposure_data, sample_type_column):
        '''Exclude all samples that are non-personal (e.g., area, etc.)'''
        exposure_data = exposure_data.copy()
        not_blank = exposure_data[sample_type_column] != 'P'
        return exposure_data.loc[~not_blank]
    #endregion

    # TODO: Consider NOT using Categorical, and switching to Parquet file.
    #region: set_categorical_dtypes
    def set_categorical_dtypes(self, exposure_data, categoricals):
        '''
        Set categorical data types for each column specified in the configuration
        settings. 

        This function is applied after loading the raw data, because Categorical
        dtypes can be challenging to read and write to a file.
        '''
        exposure_data = exposure_data.copy()
        for col, kwargs in categoricals.items():
            exposure_data[col] = pd.Categorical(exposure_data[col], **kwargs)
        return exposure_data
    #endregion

# TODO: Drop samples if unit not M or P for consistency with USIS?
# NOTE: Measure unit IDs are NaN if exposure level is null
#region: prepare_concentration_target
def prepare_concentration_target(sample_results, measure_units, mol_weights):
    '''
    Prepare the target variable of chemical concentration in air with a 
    consistent unit of mg/m3.
    '''
    return np.where(
        measure_units=='M',  # already mg/m3
        sample_results,
        np.where(
            measure_units=='P',  # ppm
            ppm_to_mg_m3(sample_results, mol_weights),
            np.nan  # everything else
        )
    )
#endregion

#region: ppm_to_mg_m3
def ppm_to_mg_m3(ppm, mw):
    '''
    Convert chemical concentration unit from parts per million to milligram 
    per cubic meter.

    Parameters
    ----------
    ppm : float or array-like
        Value in PPM.
    mw : float or array-like
        Molecular weight [g/mol].

    Reference
    ---------
    https://www.ccohs.ca/oshanswers/chemicals/convert.html
    '''
    return ppm * mw/24.45
#endregion