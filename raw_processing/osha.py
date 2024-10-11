'''
'''

import pandas as pd
import numpy as np
import json

#region: OshaDataCleaner
class OshaDataCleaner:
    '''
    A base class for cleaning OSHA-related exposure datasets.

    This class defines common cleaning operations for OSHA datasets and 
    provides a framework for running a sequence of cleaning steps dynamically.
    '''
#endregion

    # TODO: Remove short-duration samples?
    #region: clean_raw_data
    def clean_raw_data(
            self, 
            exposure_data,
            cleaning_steps, 
            log_file=None, 
            do_log_changes=True
            ):
        '''
        Clean the raw exposure data using a sequence of cleaning steps.

        Parameters
        ----------
        exposure_data : pandas.DataFrame
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
        exposure_data = exposure_data.copy()
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
    def clean_duplicates(
            self,
            exposure_data,
            unique_sample_columns,
            comparison_columns,
            substance_column
            ):
        '''
        Clean the dataset by identifying and removing duplicate samples.

        The function identifies duplicates using key columns that uniquely define 
        a sample. It removes false duplicates—records with the same key columns 
        but differing in additional columns. For true duplicates (identical 
        records), it retains one record for substance code '9010', and discards 
        duplicates for other substances.
        '''
        exposure_data = exposure_data.copy()

        ## Step 1: Identify false duplicates and remove them

        potential_duplicates = (
            exposure_data.duplicated(
                subset=unique_sample_columns, 
                keep=False
                )
        )
        false_duplicates = (
            exposure_data.loc[potential_duplicates]
            .drop_duplicates(
                subset=(
                    unique_sample_columns 
                    + comparison_columns
                ), 
                keep=False
                )
        )
        exposure_data_cleaned = exposure_data.drop(false_duplicates.index)
        
        # Step 2: Identify true duplicates and remove them selectively

        true_duplicates = (
            exposure_data_cleaned.duplicated(
                subset=unique_sample_columns, 
                keep=False
                )
        )
        
        duplicates_df = exposure_data_cleaned.loc[true_duplicates]
        non_duplicates_df = exposure_data_cleaned.loc[~true_duplicates]
        
        where_9010 = duplicates_df[substance_column] == '9010'
        duplicates_9010 = duplicates_df.loc[where_9010]
        duplicates_9010_deduped = (
            duplicates_9010.drop_duplicates(
                subset=unique_sample_columns, 
                keep='first'
                )
        )
            
        return pd.concat([non_duplicates_df, duplicates_9010_deduped])
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