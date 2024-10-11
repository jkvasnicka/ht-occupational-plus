'''
'''

import pandas as pd

from raw_processing.osha import OshaDataCleaner

#region: UsisCleaner.__init__
class UsisCleaner(OshaDataCleaner):
    '''
    A data cleaner subclass for United States Information System (USIS).

    This subclass extends the `OshaDataCleaner` to apply specific cleaning 
    methods and settings for the USIS.
    '''
    def __init__(self, path_settings, data_settings):
        self._path_settings = path_settings
        self._data_settings = data_settings
#endregion

    #region: clean_raw_data
    def clean_raw_data(
            self, 
            exposure_data, 
            do_log_changes=True
            ):
        '''
        Clean the raw exposure data using a sequence of cleaning steps.

        This method augments the corresponding method of the base class.

        Returns
        -------
        pandas.DataFrame
        '''
        exposure_data = self.pre_clean(exposure_data)

        exposure_data = super().clean_raw_data(
            exposure_data, 
            self._data_settings['cleaning_steps'],
            log_file=self._path_settings['usis_log_file'],
            do_log_changes=do_log_changes
            )

        return exposure_data
    #endregion

    # TODO: This could be common function with cehd_cleaning.py
    #region: remove_nonpersonal
    def remove_nonpersonal(self, exposure_data):
        '''
        Exclude all samples that are non-personal (e.g., area, etc.)
        '''
        exposure_data = exposure_data.copy()
        where_nonpersonal = exposure_data['sample_type_id'] != 'P'
        return exposure_data.loc[~where_nonpersonal]
    #endregion

    # TODO: Move this to separate loading as for CEHD
    #region: pre_clean
    def pre_clean(self, exposure_data):
        '''
        '''
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