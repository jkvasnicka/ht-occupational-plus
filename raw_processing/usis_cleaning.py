'''
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

    #region: prepare_clean_exposure_data
    def prepare_clean_exposure_data(self):
        '''
        '''
        exposure_data = super().prepare_clean_exposure_data(
            log_file=self.path_settings['usis_log_file']
        )
        return exposure_data
    #endregion

    #region: load_raw_data
    def load_raw_data(self):
        '''
        '''
        raw_exposure_data = usis_loading.raw_usis_data(
            self.path_settings['raw_usis_file'],
            self.data_settings['initial_dtypes']
            )
        return raw_exposure_data
    #endregion

    # FIXME: Missing identifier column in USIS prevents method inheritance
    #region: clean_duplicates
    def clean_duplicates(self, exposure_data):
        '''
        Drop duplicate samples. 

        There is an issue preventing the use of the base class method. The 
        USIS dataset appears to lack a column that identifies a unique sample
        like "field number" in the CEHD. The U of M documentation indicates 
        that there is a "measure ID" column, but this column appears to be 
        missing. Perhaps create a GitHub issue. If this column were available,
        then unique samples could be identified and the parent method could be 
        directly inherited.
        '''
        return exposure_data.drop_duplicates()
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