'''
This module defines the UsisCleaner class, providing a data cleaning 
methodology specific to the USIS dataset by OSHA.
'''

from . import usis_loading
from raw_processing.osha_cleaning import OshaDataCleaner

#region: UsisCleaner.__init__
class UsisCleaner(OshaDataCleaner):
    '''
    A data cleaner subclass for United States Information System (USIS).

    This subclass extends the `OshaDataCleaner` to apply specific cleaning 
    methods and settings for the USIS.
    '''
    def __init__(self, data_settings, path_settings, comptox_settings=None):
        super().__init__(data_settings, path_settings, comptox_settings)
#endregion

    #region: clean_exposure_data
    def clean_exposure_data(self):
        '''
        Main data cleaning function.

        Wrapper around the parent class method, specifies the log file path.

        Returns
        -------
        pandas.DataFrame
            Cleaned dataset.
        '''
        exposure_data = super().clean_exposure_data(
            log_file=self.path_settings['usis_log_file']
        )
        return exposure_data
    #endregion

    #region: load_raw_data
    def load_raw_data(self):
        '''
        Loads the raw USIS dataset using the current config.
        '''
        raw_exposure_data = usis_loading.raw_usis_data(
            self.path_settings['raw_usis_file'],
            self.data_settings['initial_dtypes']
            )
        return raw_exposure_data
    #endregion

    #region: clean_duplicates
    def clean_duplicates(self, exposure_data):
        '''
        Clean the dataset by identifying and removing duplicate samples.

        1. Removes any exact duplicates across all columns, retaining the 
           first occurrence. 
        2. Removes any duplicates across the subset of columns that 
           corresponds to a unique sample, retaining the first occurrence.
                a) If there is a mix of different measurement types in the 
                dataset (e.g., short-term vs. TWA, personal vs. area), 
                then the subset of columns is dynamically adjusted to ensure 
                that records corresponding to different sample types are not
                treated as duplicates.
                b) If there is NOT a mix of different measurement types, then
                the subset of columns defines a single worker. Here, it is 
                assumed that a worker should only have one sample for a given
                chemical and inspection (e.g., there should not be several 
                full-shift TWA measurements).

        Notes
        -----
        - The 'chem_id_col' (e.g., DTXSID), if present, is used to define a 
          single substance. This is because there may be several IMIS 
          substance codes ('substance_code_col') for a single substance, based
          on how OSHA defines these codes. For example, Formaldehyde may have 
          substance codes 1290 ('Formaldehyde') and 1291 
          ['FORMALDEHYDE (ACTION LEVEL)']. If the 'chem_id_col' is not 
          present, then the susbtance code is used instead.
        '''
        exposure_data = exposure_data.copy()

        # Define columns that should describe a unique sample for a worker
        if self.comptox_settings is not None:
            chem_id_col = self.comptox_settings.get('chem_id_col')
        else:
            chem_id_col = self.data_settings.get('substance_code_col')

        unique_sample_cols = [
            chem_id_col,
            self.data_settings['naics_code_col'],
            self.data_settings['inspection_number_col'],
            self.data_settings['sampling_number_col']
        ]

        # Remove exact duplicates across all columns
        # The first occurrence is retained
        exposure_data = exposure_data.drop_duplicates() 

        # Dynamically adjust 'unique_sample_cols' so that different 
        # measurement types, if present, are not considered as duplicates
        if len(set(exposure_data['exposure_type_id'])) > 1:
            # A worker may have several exposure-type samples (e.g., TWA, ...)
            unique_sample_cols += ['exposure_type_id']
        if len(set(exposure_data['sample_type_id'])) > 1:
            # A worker may have several sample-type samples (e.g., area, ...)
            unique_sample_cols += ['sample_type_id']

        return exposure_data.drop_duplicates(subset=unique_sample_cols)
    #endregion

    #region: remove_nonpersonal
    def remove_nonpersonal(self, exposure_data):
        '''Exclude all samples that are non-personal (e.g., area, etc.)'''
        return super().remove_nonpersonal(exposure_data, 'sample_type_id')
    #endregion

    #region: remove_non_full_shift_twa
    def remove_non_full_shift_twa(self, exposure_data):
        '''
        Remove short-term samples, etc.
        '''
        exposure_data = exposure_data.copy()
        where_full_shift_twa = exposure_data['exposure_type_id'] == 'T'
        exposure_data = exposure_data.loc[where_full_shift_twa]
        return exposure_data
    #endregion