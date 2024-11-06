'''
'''

import pandas as pd

from raw_processing.osha_cleaning import OshaDataCleaner

#region: UsisCleaner.__init__
class UsisCleaner(OshaDataCleaner):
    '''
    A data cleaner subclass for United States Information System (USIS).

    This subclass extends the `OshaDataCleaner` to apply specific cleaning 
    methods and settings for the USIS.
    '''
    def __init__(self, data_settings, path_settings):
        super().__init__(data_settings, path_settings)
#endregion

    #region: clean_raw_data
    def clean_raw_data(
            self, 
            raw_exposure_data, 
            do_log_changes=True
            ):
        '''
        Clean the raw exposure data using a sequence of cleaning steps.

        This method augments the corresponding method of the base class.

        Returns
        -------
        pandas.DataFrame
        '''
        exposure_data = super().clean_raw_data(
            raw_exposure_data, 
            self.data_settings['cleaning_steps'],
            log_file=self.path_settings['usis_log_file'],
            do_log_changes=do_log_changes
            )

        return exposure_data
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

#region: load_raw_usis_data
def load_raw_usis_data(raw_usis_file):
    '''
    '''
    exposure_data = pd.read_feather(raw_usis_file)
    # Apply minimal data cleaning for consistency with CEHD
    return pre_clean(exposure_data)
#endregion

#region: pre_clean
def pre_clean(exposure_data):
    '''Apply minimal data cleaning for consistency with CEHD'''
    exposure_data = exposure_data.copy()
    exposure_data['year'] = _extract_sample_year(exposure_data['sample_date'])
    return exposure_data
#endregion

#region: _extract_sample_year
def _extract_sample_year(sample_dates):
    '''
    Extract the sample year from the sample date.

    Creates a new column with the sample year.
    '''
    return (
        pd.to_datetime(sample_dates)
        .dt.year
        .astype('int64')
    )
#endregion