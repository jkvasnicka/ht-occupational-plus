'''
'''

import pandas as pd
import json
import os

from . import comptox

#region: OshaDataCleaner.__init__
class OshaDataCleaner:
    '''
    A base class for cleaning OSHA-related exposure datasets.

    This class defines common cleaning operations for OSHA datasets and 
    provides a framework for running a sequence of cleaning steps dynamically.
    '''
    def __init__(self, data_settings, path_settings, comptox_settings=None):
        self.data_settings = data_settings
        self.path_settings = path_settings
        self.comptox_settings = comptox_settings
#endregion

    #region: prepare_clean_exposure_data
    def prepare_clean_exposure_data(self, log_file=None):
        '''
        Provides the main interface. 
        
        Encapsulates raw data loading and cleaning.
        '''
        raw_exposure_data = self.load_raw_data()

        exposure_data = self.clean_raw_data(
            raw_exposure_data,
            log_file=log_file
            )
        
        return exposure_data
    #endregion

    #region: load_raw_data
    def load_raw_data(self):
        '''To be defined in the subclass.'''
        pass
    #endregion

    #region: clean_raw_data
    def clean_raw_data(self, raw_exposure_data, log_file=None):
        '''
        Clean the raw exposure data using a sequence of cleaning steps.

        Returns
        -------
        pandas.DataFrame
            The cleaned exposure data.
        '''
        exposure_data = raw_exposure_data.copy()
        change_log = {}  # initialize

        exposure_data = self.set_categorical_dtypes(
            exposure_data, 
            self.data_settings.get('categoricals', {})
            )
        
        for step_name in self.data_settings['cleaning_steps']:
            N_before = len(exposure_data)
            # Dynamically get the cleaning method from the step name
            exposure_data = getattr(self, step_name)(exposure_data)
            N_after = len(exposure_data)
            change_log[step_name] =  N_after - N_before

        if log_file is not None:
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

        unique_sample_cols = self.data_settings['unique_sample_cols']
        comparison_cols = self.data_settings['comparison_cols']
        substance_code_col = self.data_settings['substance_code_col']

        ## Step 1: Identify and remove conflicting duplicates

        where_unique_sample_duplicate = (
            exposure_data.duplicated(
                subset=unique_sample_cols, 
                keep=False
                )
        )
        conflicting_samples = (
            exposure_data.loc[where_unique_sample_duplicate]
            .drop_duplicates(
                subset=(
                    unique_sample_cols 
                    + comparison_cols
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
                subset=unique_sample_cols, 
                keep='first'
                )
        )
            
        return pd.concat([non_duplicates_data, duplicates_9010_deduped])
    #endregion

    #region: remove_nonpersonal
    def remove_nonpersonal(self, exposure_data, sample_type_col):
        '''Exclude all samples that are non-personal (e.g., area, etc.)'''
        exposure_data = exposure_data.copy()
        not_blank = exposure_data[sample_type_col] != 'P'
        return exposure_data.loc[~not_blank]
    #endregion

    #region: convert_to_mass_concentration
    def convert_to_mass_concentration(self, exposure_data):
        '''
        Convert samples results to mass concentration (mg/m³) where applicable

        All other samples types are filtered out.
        '''
        exposure_data = exposure_data.copy()

        measure_unit_col = self.data_settings['measure_unit_col']
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

        chem_id_col = self.comptox_settings['chem_id_col']
        input_col = self.comptox_settings['input_col']
        molecular_weight_col = self.comptox_settings['molecular_weight_col']
        measure_unit_col = self.data_settings['measure_unit_col']
        sample_result_col = self.data_settings['sample_result_col']
        substance_name_col = self.data_settings['substance_name_col']
        
        comptox_data = comptox.data_from_raw(
            self.path_settings['comptox_file'], 
            self.comptox_settings['found_by_col'], 
            self.comptox_settings['duplicate_warning'], 
            self.comptox_settings['rank_map'], 
            input_col
        )

        # NOTE: Assumes the input is the substance name
        chem_id_for_name = (
            comptox_data
            .set_index(input_col)[chem_id_col]
            .to_dict()
        )
        mw_for_chem_id = (
            comptox_data
            .set_index(chem_id_col)[molecular_weight_col]
            .to_dict()
        )

        where_ppm = exposure_data[measure_unit_col] == 'P'
        ppm_values = exposure_data.loc[where_ppm, sample_result_col]
        mol_weights = (
            exposure_data.loc[where_ppm, substance_name_col]
            .map(chem_id_for_name)
            .map(mw_for_chem_id)
        )

        # Convert units only where molecular weights are available
        # This avoids propagating NaNs to the calculated result
        where_valid_mw = mol_weights.notna()
        mg_m3_values = ppm_to_mg_m3(
            ppm_values.loc[where_valid_mw], 
            mol_weights.loc[where_valid_mw]
            )
        where_to_convert = where_ppm & where_valid_mw
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

        measure_units = exposure_data[self.data_settings['measure_unit_col']]
        sample_results = (
            exposure_data[self.data_settings['sample_result_col']]
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

        Samples that cannot be identified are removed.
        '''
        exposure_data = exposure_data.copy()

        input_col = self.comptox_settings['input_col']
        chem_id_col = self.comptox_settings['chem_id_col']

        comptox_data = comptox.data_from_raw(
            self.path_settings['comptox_file'], 
            self.comptox_settings['found_by_col'], 
            self.comptox_settings['duplicate_warning'], 
            self.comptox_settings['rank_map'], 
            input_col
        )

        chem_id_for_name = (
            comptox_data
            .set_index(input_col)[chem_id_col]
            .to_dict()
        )
        exposure_data[chem_id_col] = (
            exposure_data[self.data_settings['substance_name_col']]
            .map(chem_id_for_name)
        )

        # Remove unidentifiable samples
        return exposure_data.dropna(subset=chem_id_col)
    #endregion

    #region: harmonize_naics_codes
    def harmonize_naics_codes(self, exposure_data):
        '''
        Harmonize NAICS codes in the given dataset to the latest standard.

        Samples with missing NAICS codes post-harmonization are removed.
        '''
        exposure_data = exposure_data.copy()

        exposure_data = self._convert_sic_to_naics(exposure_data)
        exposure_data = self._update_naics_to_latest(exposure_data)
        exposure_data = self._remove_missing_naics(exposure_data)

        return exposure_data
    #endregion

    #region:_convert_sic_to_naics
    def _convert_sic_to_naics(self, exposure_data):
        '''
        Convert SIC codes to NAICS codes where NAICS codes are missing.
        '''
        sic_conversion_file = self.path_settings['sic_conversion_file']
        sic_code_col = self.data_settings['sic_code_col']
        naics_code_col = self.data_settings['naics_code_col']

        naics_for_sic = load_concordance_table(sic_conversion_file, header=1)

        where_to_convert = (
            exposure_data[naics_code_col].isna() 
            & exposure_data[sic_code_col].notna()
        )
        exposure_data.loc[where_to_convert, naics_code_col] = (
            exposure_data.loc[where_to_convert, sic_code_col]
            .map(naics_for_sic)
        )

        return exposure_data
    #endregion

    #region: _update_naics_to_latest
    def _update_naics_to_latest(self, exposure_data):
        '''
        Update NAICS codes to the latest standard using concordance tables.

        This method sequentially applies NAICS-to-NAICS concordance tables to 
        update all NAICS codes in the dataset to the most recent NAICS cycle. 
        It assumes the concordance tables are named to reflect the years of 
        each transition (e.g., "2002_to_2007_NAICS.xls") and stored in a 
        dedicated directory.
        '''
        naics_concordances_dir = self.path_settings['naics_concordances_dir']
        naics_code_col = self.data_settings['naics_code_col']
        
        file_names = sort_files_by_year(os.listdir(naics_concordances_dir))

        # Initialize the full mapping 
        first_file = os.path.join(naics_concordances_dir, file_names[0])
        full_mapping = load_concordance_table(first_file)

        # Update the mapping with each subsequent NAICS cycle
        for file_name in file_names[1:]:
            next_file = os.path.join(naics_concordances_dir, file_name)
            next_mapping = load_concordance_table(next_file)
            # If no mapping is found, retain the original value
            full_mapping = {
                k : next_mapping.get(v, v) 
                for k, v in full_mapping.items()
                }

        # Replace original NAICS codes with those from the latest cycle
        exposure_data.loc[:, naics_code_col] = (
            exposure_data[naics_code_col].map(full_mapping)
        )

        return exposure_data
    #endregion

    #region: _remove_missing_naics
    def _remove_missing_naics(self, exposure_data):
        '''Remove samples with missing NAICS code'''
        exposure_data = exposure_data.copy()
        naics_code_col = self.data_settings['naics_code_col']
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

#region: load_concordance_table
def load_concordance_table(
        concordance_file, 
        header=2, 
        old_col_idx=0, 
        new_col_idx=2, 
        dtype=None
        ):
    '''
    Load a NAICS concordance table from an Excel file and create a mapping 
    from old NAICS codes to new NAICS codes.

    Notes
    -----
    - Many-to-One (Aggregation): This case is handled correctly, as multiple 
      old NAICS codes pointing to the same new code will all map to that new 
      code.
    - One-to-Many (Splitting): In cases where an old NAICS code splits into 
      several new codes, only the last encountered new code will be retained 
      in the mapping. This behavior is due to the overwriting nature of Python
      dictionaries.
    '''        
    # TODO: Harmonize with the exposure_data dtypes
    if dtype is None:
        dtype = 'str'

    df = pd.read_excel(concordance_file, header=header)

    old_values = df[df.columns[old_col_idx]].astype(dtype)
    new_values = df[df.columns[new_col_idx]].astype(dtype)

    return dict(zip(old_values, new_values))
#endregion

#region: sort_files_by_year
def sort_files_by_year(file_names):
    '''Sort file names chronologically based on the starting year.'''
    extract_year = lambda x: x.split('_')[0]
    
    for file in file_names:
        year = extract_year(file)
        if not year.isdigit() or len(year) != 4:
            raise ValueError(
                'Not a valid file name (does not begin with a valid year): '
                f'{file}')
            
    return sorted(file_names, key=lambda x: int(extract_year(x)))
#endregion

#region: load_json
def load_json(file):
    '''This can be used to load the change log from a JSON file.'''
    with open(file, 'r') as file:
        json_dict = json.load(file)
    return json_dict
#endregion