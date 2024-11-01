'''
'''

import pandas as pd
import json

#region: OshaDataCleaner.__init__
class OshaDataCleaner:
    '''
    A base class for cleaning OSHA-related exposure datasets.

    This class defines common cleaning operations for OSHA datasets and 
    provides a framework for running a sequence of cleaning steps dynamically.
    '''
    def __init__(self, data_settings, path_settings):
        self._data_settings = data_settings
        self._path_settings = path_settings
#endregion

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
        substance_code_col = self._data_settings['substance_code_col']

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
        where_9010 = duplicates_data[substance_code_col] == '9010'
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

    #region: convert_to_mass_concentration
    def convert_to_mass_concentration(self, exposure_data):
        '''
        Convert samples results to mass concentration (mg/m³) where applicable

        All other samples types are filtered out.
        '''
        exposure_data = exposure_data.copy()

        measure_unit_col = self._data_settings['measure_unit_col']
        # FIXME: Temporary workaround while Categorical types are being used
        # Avoids TypeError: Cannot setitem on a Categorical ...
        exposure_data[measure_unit_col] = (
            exposure_data[measure_unit_col].astype('object')
        )

        exposure_data = self._convert_ppm_to_mg_m3(exposure_data)
        exposure_data = self._convert_percent_to_mg_m3(exposure_data)
        exposure_data = self._remove_non_mg_m3_units(exposure_data)

        return exposure_data
    #endregion

    #region: _convert_ppm_to_mg_m3
    def _convert_ppm_to_mg_m3(self, exposure_data):
        '''
        Convert sample results from parts per million (PPM) to mass 
        concentration (mg/m³).
        '''
        exposure_data = exposure_data.copy()

        chem_id_file = self._path_settings['chem_id_file']
        measure_unit_col = self._data_settings['measure_unit_col']
        sample_result_col = self._data_settings['sample_result_col']
        substance_name_col = self._data_settings['substance_name_col']

        # TODO: Move these strings to config?
        chem_id_for_name = mapping_from_chem_id_file(
            chem_id_file, 
            'INPUT', 
            'DTXSID'
            )
        mw_for_chem_id = mapping_from_chem_id_file(
            chem_id_file, 
            'DTXSID', 
            'AVERAGE_MASS'
            )

        where_to_convert = exposure_data[measure_unit_col] == 'P'  # PPM
        ppm_values = exposure_data.loc[where_to_convert, sample_result_col]
        mol_weights = (
            exposure_data.loc[where_to_convert, substance_name_col]
            .map(chem_id_for_name)
            .map(mw_for_chem_id)
        )

        mg_m3_values = ppm_to_mg_m3(ppm_values, mol_weights)
        exposure_data.loc[where_to_convert, sample_result_col] = mg_m3_values
        exposure_data.loc[where_to_convert, measure_unit_col] = 'M_from_PPM'

        return exposure_data
    #endregion

    #region: _convert_percent_to_mg_m3
    def _convert_percent_to_mg_m3(self, exposure_data):
        '''
        Convert sample results from percentage to mass concentration (mg/m³).

        Placeholder to be defined in the subclass if applicable.
        '''
        return exposure_data
    #endregion

    #region: _remove_non_mg_m3_units
    def _remove_non_mg_m3_units(self, exposure_data):
        '''
        Remove samples with non-mass-concentration unit of measurement.
        '''
        exposure_data = exposure_data.copy()

        measure_units = exposure_data[self._data_settings['measure_unit_col']]
        sample_results = (
            exposure_data[self._data_settings['sample_result_col']]
        )

        # Match strings starting with 'M' followed by '_' or end of string
        # Retain samples where measure unit is NaN and result is non-detect
        where_mass_conc = (
            measure_units.str.contains(r'^M(?:_|$)')
            | (measure_units.isna() & (sample_results == 0.))
        )
        exposure_data = exposure_data.loc[where_mass_conc]

        return exposure_data
    #endregion

    #region: convert_substance_names_to_ids
    def convert_substance_names_to_ids(self, exposure_data):
        '''
        Insert a new column of DSSTox chemical identifiers corresponding to 
        the substance names. 

        Drop samples with missing identifiers.
        '''
        exposure_data = exposure_data.copy()

        chem_id_for_name = mapping_from_chem_id_file(
            self._path_settings['chem_id_file'], 
            'INPUT', 
            'DTXSID'
        )
        exposure_data['DTXSID'] = (
            exposure_data[self._data_settings['substance_name_col']]
            .map(chem_id_for_name)
        )

        return exposure_data.dropna(subset='DTXSID')
    #endregion

    #region: remove_missing_naics
    def remove_missing_naics(self, exposure_data):
        '''Remove samples with missing NAICS code'''
        exposure_data = exposure_data.copy()
        naics_code_col = self._data_settings['naics_code_col']
        return exposure_data.dropna(subset=naics_code_col)
    #endregion

    # TODO: Consider NOT using Categorical, and switching to Parquet file.
    # Only temporarily set Categorical where it's needed for an operation
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

#region: mapping_from_chem_id_file
def mapping_from_chem_id_file(chem_id_file, key_col, value_col):
    '''
    Generate a mapping based on the inputted chemical ID file and column 
    names.
    '''
    mapping = (
        pd.read_csv(chem_id_file)
        .set_index(key_col)[value_col]
        .dropna()
        .to_dict()
    )
    return mapping
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

    Notes
    -----
    24.45 is the volume in liters occupied by a mole of air at 25ºC and 760 
    torr.

    This formula can be used when measurements are taken at 25°C and the air 
    pressure is 760 torr (= 1 atmosphere or 760 mm Hg).

    Reference
    ---------
    https://www.ccohs.ca/oshanswers/chemicals/convert.html
    '''
    return ppm * mw/24.45
#endregion