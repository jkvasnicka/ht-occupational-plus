'''
This module implements a data cleaning methodology developed by Prof. Jérôme 
Lavoué (Université de Montréal) et al. for the Chemical Exposure Health Data 
(CEHD).

### Context and Purpose

The original data cleaning pipeline was written in R 
(CEHD cleaning 1984_2018.R) and has been translated into Python to ensure 
consistency with the initial implementation while integrating it into our 
broader Python-based workflow. This translation was performed nearly verbatim 
to maintain fidelity to the original logic and thereby facilitate testing and
debugging. Consequently, the Python code may not follow best practices in 
software engineering.  

### Remaining Discrepancies Between R and Python Outputs

During the translation, several discrepancies between the R and Python outputs
were identified using pandas.testing.assert_series_equal():

1. Bugs in the Original R Code: Some issues in the original R implementation, 
   such as hardcoded values or incorrect sequence generation, were identified 
   and corrected in the Python version.

2. Data Formatting Differences: Remaining discrepancies arose due to 
   differences in how R and Python handle data formatting. These discrepancies
   were deemed to be insignificant (e.g., extra double quotes in R).
'''

import pandas as pd
import os

from raw_processing import cehd_loading
from raw_processing.osha_cleaning import OshaDataCleaner

#region: CehdCleaner.__init__
class CehdCleaner(OshaDataCleaner):
    '''
    A data cleaner subclass for Chemical Exposure Health Data (CEHD).

    This subclass extends the `OshaDataCleaner` to apply specific cleaning 
    methods and settings for the CEHD.
    '''
    def __init__(self, data_settings, path_settings, comptox_settings=None):
        super().__init__(data_settings, path_settings, comptox_settings)

        # Apply CEHD-specific initialization
        self._qualif_conv_2020 = load_qualifier_conversion(
            path_settings['qualif_conv_file']
            )
        self._unit_conv_2020 = load_unit_measure_conversion(
            path_settings['unit_conv_file']
            )
#endregion

    #region: clean_exposure_data
    def clean_exposure_data(self):
        '''
        '''
        exposure_data = super().clean_exposure_data(
            log_file=self.path_settings['cehd_log_file']
        )
        return exposure_data
    #endregion

    #region: load_raw_data
    def load_raw_data(self):
        '''
        '''
        raw_exposure_data = cehd_loading.raw_chem_exposure_health_data(
            self.data_settings, 
            self.path_settings
            )
        return raw_exposure_data
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
    
    #region: clean_instrument_type
    def clean_instrument_type(self, exposure_data):
        '''
        Comprehensive function to handle the cleaning of instrument type.
        '''
        exposure_data = self._handle_missing_instrument_type(exposure_data)
        table_for_subs = self.load_instrument_type_tables()
        exposure_data = (
            self._apply_instrument_type_tables(exposure_data, table_for_subs)
        )
        exposure_data = self._handle_remaining_missing_instrument_type(exposure_data)
        return exposure_data
    #endregion

    #region: _handle_missing_instrument_type
    def _handle_missing_instrument_type(self, exposure_data):
        '''
        Handle missing instrument type and perform initial population and cleanup.
        '''
        where_nan = exposure_data['INSTRUMENT_TYPE'].isna()
        exposure_data.loc[where_nan, 'INSTRUMENT_TYPE'] = ''

        # For years before 2012, keep original data
        where_pre_2012 = exposure_data['YEAR'].astype(int) < 2012
        exposure_data.loc[~where_pre_2012, 'INSTRUMENT_TYPE'] = 'not recorded'

        return exposure_data
    #endregion

    #region: load_instrument_type_tables
    def load_instrument_type_tables(self):
        '''
        Load IT tables for each substance code.
        '''
        it_directory = self.path_settings['it_directory']

        csv_files = [f for f in os.listdir(it_directory) if f.endswith('.csv')]
        table_for_subs = {}

        for file in csv_files:
            # Extract the substance code from the filename
            subs_code = file[2:6]
            # NOTE: Missing values are replaced with '' like R's read.csv
            df = pd.read_csv(os.path.join(it_directory, file), sep=',').fillna('')
            table_for_subs[subs_code] = df

        return table_for_subs
    #endregion

    #region: _apply_instrument_type_tables
    def _apply_instrument_type_tables(self, exposure_data, table_for_subs):
        '''
        Clean instrument type for specific substance codes using conversion
        tables.
        '''
        exposure_data = exposure_data.copy()

        for subs_code, it_table in table_for_subs.items():
            # For each clean values, get the corresponding raw value(s)
            for clean_value in it_table['clean'].unique():
                where_clean_value = it_table['clean'] == clean_value
                raw_values_to_clean = list(
                    it_table.loc[where_clean_value, 'raw'].astype('str')
                    )
                
                where_to_clean = (
                    (exposure_data['IMIS_SUBSTANCE_CODE'] == subs_code) 
                    & (exposure_data['YEAR'].astype(int) < 2010)
                    & (exposure_data['INSTRUMENT_TYPE'].isin(raw_values_to_clean))
                    )
                exposure_data.loc[where_to_clean, 'INSTRUMENT_TYPE'] = (
                    clean_value
                )

        return exposure_data
    #endregion

    #region: _handle_remaining_missing_instrument_type
    def _handle_remaining_missing_instrument_type(self, exposure_data):
        '''
        Final cleanup for 'INSTRUMENT_TYPE'.

        Sets empty strings to 'eliminate' and removes all samples designated as 
        'eliminate', including those set through conversion tables.
        '''
        exposure_data = exposure_data.copy()
        where_empty = exposure_data['INSTRUMENT_TYPE'] == ''
        exposure_data.loc[where_empty, 'INSTRUMENT_TYPE'] = 'eliminate'
        rows_to_exclude = exposure_data['INSTRUMENT_TYPE'] == 'eliminate'
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_zero_volume_sampled
    def remove_zero_volume_sampled(self, exposure_data):
        '''
        Remove samples that have an air volume sampled of zero.
        '''
        exposure_data = exposure_data.copy()
        rows_to_exclude = exposure_data['AIR_VOLUME_SAMPLED'] == 0.
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_missing_volume
    def remove_missing_volume(self, exposure_data):
        '''
        Remove samples that have a missing or empty volume sampled variable.

        This function identifies and removes samples where the 'AIR_VOLUME_SAMPLED'
        column is either missing (NaN) or an empty string ('').
        '''
        exposure_data = exposure_data.copy()
        
        rows_to_exclude = (
            exposure_data['AIR_VOLUME_SAMPLED'].isna() 
            | (exposure_data['AIR_VOLUME_SAMPLED'] == '')
        )

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    # NOTE: Inconsistency
    #region: remove_missing_sample_number
    def remove_missing_sample_number(self, exposure_data):
        '''
        Remove samples that have a missing or null sampling number.
        
        Note:
        - In the original R script, '0' and '0.0' were treated as distinct values 
        because they were stored as strings. However, in Python, when converting 
        to numeric , both '0' and '0.0' are treated as numeric zero (0.0) and 
        thus identified as null values by this function.
        '''
        exposure_data = exposure_data.copy()
        
        rows_to_exclude = (
            exposure_data['SAMPLING_NUMBER'].isna() 
            | (pd.to_numeric(
                exposure_data['SAMPLING_NUMBER'], errors='coerce') == 0.
                )
        )

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_negative_sample_result
    def remove_negative_sample_result(self, exposure_data):
        '''
        Remove samples with a sample result less than zero.
        '''
        exposure_data = exposure_data.copy()
        rows_to_exclude = exposure_data['SAMPLE_RESULT'] < 0.
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_null_time_sampled
    def remove_null_time_sampled(self, exposure_data):
        '''
        Remove samples that have a null time sampled variable.
        '''
        exposure_data = exposure_data.copy()
        rows_to_exclude = exposure_data['TIME_SAMPLED'] == 0.
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_missing_time_sampled
    def remove_missing_time_sampled(self, exposure_data):
        '''
        Remove samples that have a missing value for the time sampled variable.
        '''
        exposure_data = exposure_data.copy()
        rows_to_exclude = exposure_data['TIME_SAMPLED'].isna()
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_missing_office_identifier
    def remove_missing_office_identifier(self, exposure_data):
        '''
        Remove samples that have a missing value for the office ID.
        '''
        exposure_data = exposure_data.copy()
        rows_to_exclude = exposure_data['OFFICE_ID'].isna()
        return exposure_data.loc[~rows_to_exclude]
    #endregion
    
    # FIXME: Double check conversion factor. Unclear.
    #region: _convert_percent_to_mg_m3
    def _convert_percent_to_mg_m3(self, exposure_data):
        '''
        Convert sample results from percentage to mass concentration (mg/m³).
        '''
        exposure_data = exposure_data.copy()

        exposure_data = self.remove_null_weight(exposure_data)

        where_to_convert = (
            (exposure_data['SAMPLE_WEIGHT'] != 0) 
            & (exposure_data['UNIT_OF_MEASUREMENT'] == '%') 
            & (exposure_data['SAMPLE_RESULT'] > 0) 
            & exposure_data['SAMPLE_WEIGHT'].notna() 
            & exposure_data['AIR_VOLUME_SAMPLED'].notna() 
            & (exposure_data['AIR_VOLUME_SAMPLED'] > 0)
        )

        sample_result = exposure_data.loc[where_to_convert, 'SAMPLE_RESULT']
        sample_weight = exposure_data.loc[where_to_convert, 'SAMPLE_WEIGHT']
        air_volume_sampled = (
            exposure_data.loc[where_to_convert, 'AIR_VOLUME_SAMPLED']
        )

        conversion_factor = 10.

        converted_result = (
            (sample_result * sample_weight * conversion_factor) 
            / air_volume_sampled
        )

        # Assign the converted results back to the dataframe
        exposure_data.loc[where_to_convert, 'SAMPLE_RESULT'] = converted_result
        exposure_data.loc[where_to_convert, 'UNIT_OF_MEASUREMENT'] = (
            'M_from_Perc'
            )
        
        return exposure_data
    #endregion

    #region: remove_null_weight
    def remove_null_weight(self, exposure_data):
        '''
        Remove samples where unit of measurement is percentage ('%'), the sample 
        result is non-null, but the sample weight is null.
        '''
        exposure_data = exposure_data.copy()

        # TODO: Is this step necessary?
        exposure_data['SAMPLE_WEIGHT'] = (
            exposure_data['SAMPLE_WEIGHT'].fillna(0)
        )

        rows_to_exclude = (
            (exposure_data['SAMPLE_WEIGHT'] == 0) &
            (exposure_data['UNIT_OF_MEASUREMENT'] == '%') &
            (exposure_data['SAMPLE_RESULT'] > 0)
        )

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    # TODO: Remove hardcoding?
    #region: remove_invalid_unit
    def remove_invalid_unit(self, exposure_data):
        '''
        For each list of substance codes, remove samples where the unit of
        measurement is invalid.
        '''
        top_substances = [
            '0040', '0230', '0260', '0360', '0430', '0491', '0685', 
            '0720', '0731', '1073', '1290', '1520', '1560', '1591', 
            '1620', '1730', '1790', '1840', '2270', '2280', '2460', 
            '2571', '2590', '2610', '9020', '9130', '9135', 'C141', 'S103'
        ]
        # TODO: Should empty string be NaN?
        valid_units_n31 = ['', 'F', 'P', 'M']
        exposure_data = self._remove_invalid_unit_for_substance_codes(
            exposure_data, 
            top_substances, 
            valid_units_n31
            )

        valid_units_n32 = ['', '%', 'M']
        exposure_data = self._remove_invalid_unit_for_substance_codes(
            exposure_data, 
            ['9010'], 
            valid_units_n32
            )

        where_other = (
            ~exposure_data['IMIS_SUBSTANCE_CODE'].isin(top_substances + ['9010'])
            )
        other_substances = list(
            exposure_data.loc[where_other, 'IMIS_SUBSTANCE_CODE'].unique()
            )
        valid_units_n33 = ['', '%', 'M', 'P', 'F']
        exposure_data = self._remove_invalid_unit_for_substance_codes(
            exposure_data, 
            other_substances, 
            valid_units_n33
            )

        return exposure_data
    #endregion

    #region: _remove_invalid_unit_for_substance_codes
    def _remove_invalid_unit_for_substance_codes(
            self,
            exposure_data, 
            substance_codes, 
            valid_units
            ):
        '''
        Remove samples where the unit of measurement is invalid for given
        substances.
        '''
        exposure_data = exposure_data.copy()

        where_in_substance_codes = (
            exposure_data['IMIS_SUBSTANCE_CODE'].isin(substance_codes)
        )
        where_invalid_units = (
            ~exposure_data['UNIT_OF_MEASUREMENT'].isin(valid_units)
        )

        rows_to_exclude = where_in_substance_codes & where_invalid_units
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_percent_greater_than_100
    def remove_percent_greater_than_100(self, exposure_data):
        '''
        Remove samples where the unit of measurement is '%' and the sample result
        is greater than 100.
        '''
        rows_to_exclude = (
            (exposure_data['UNIT_OF_MEASUREMENT'] == '%') &
            (exposure_data['SAMPLE_RESULT'] > 100.)
        )
        
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_empty_unit_non_null_result
    def remove_empty_unit_non_null_result(self, exposure_data):
        '''
        Remove samples where the unit of measurement is empty and the sample 
        result is not null.
        '''
        rows_to_exclude = (
            (exposure_data['UNIT_OF_MEASUREMENT'] == '') &
            (exposure_data['SAMPLE_RESULT'] > 0)
        )
        
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_invalid_fibers_unit
    def remove_invalid_fibers_unit(self, exposure_data):
        '''
        Remove samples with specific substance codes that should not have "F" as
        the unit of measurement.
        '''
        # These codes should not have "F" as the unit of measurement
        non_f_substance_codes = self.data_settings['non_f_substance_codes']
        
        where_invalid_units = (
            (exposure_data['UNIT_OF_MEASUREMENT'] == 'F') &
            (exposure_data['IMIS_SUBSTANCE_CODE'].isin(non_f_substance_codes))
        )
        
        return exposure_data.loc[~where_invalid_units]
    #endregion

    #region: remove_qualifier_unit_mismatch
    def remove_qualifier_unit_mismatch(self, exposure_data):
        '''
        Remove samples with inconsistent qualifier and unit of measurement.
        '''
        exposure_data = exposure_data.copy()

        condition_inconsistent_units = (
            (exposure_data['UNIT_OF_MEASUREMENT'] != '%') 
            & (exposure_data['QUALIFIER'] == '%')
        ) | (
            (exposure_data['UNIT_OF_MEASUREMENT'] != 'M') 
            & (exposure_data['QUALIFIER'] == 'M')
        )

        return exposure_data.loc[~condition_inconsistent_units]
    #endregion

    #region: remove_approximate_measure
    def remove_approximate_measure(self, exposure_data):
        '''
        Remove samples where the QUALIFIER indicates an approximate measure.
        '''
        exposure_data = exposure_data.copy()
        approximate_qualifiers = self.data_settings['approximate_qualifiers']

        rows_to_exclude = exposure_data['QUALIFIER'].isin(approximate_qualifiers)
        
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_yttrium_substance_conflict
    def remove_yttrium_substance_conflict(self, exposure_data):
        '''
        Remove samples where the qualifier 'Y' is used but the substance code is
        not 9135.
        '''
        exposure_data = exposure_data.copy()

        rows_to_exclude = (
            (exposure_data['QUALIFIER'] == 'Y') 
            & (exposure_data['IMIS_SUBSTANCE_CODE'] != '9135')
        )

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_fibers_substance_conflict
    def remove_fibers_substance_conflict(self, exposure_data):
        '''
        Remove samples where the qualifier suggests fibers (F) but the substance
        code is not 9020.
        '''
        exposure_data = exposure_data.copy()
        rows_to_exclude = (
            (exposure_data['QUALIFIER'] == 'F') 
            & (exposure_data['IMIS_SUBSTANCE_CODE'] != '9020')
        )
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_combustion_related
    def remove_combustion_related(self, exposure_data):
        '''
        Remove samples with qualifiers related to combustion.
        '''
        exposure_data = exposure_data.copy()
        combustion_qualifiers = self.data_settings['combustion_qualifiers']

        rows_to_exclude = exposure_data['QUALIFIER'].isin(combustion_qualifiers)

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_blk_possible_bulk_not_blank
    def remove_blk_possible_bulk_not_blank(self, exposure_data):
        '''
        Remove samples judged to be possible blank (BLK) and bulk, yet BLANK_USED
        is 'N'.
        '''
        exposure_data = exposure_data.copy()

        condition_blk_possible_bulk = (
            (self._qualif_conv_2020['clean'] == 'BLK')
            & (self._qualif_conv_2020['possible_bulk'] == 'Y')
        )
        rows_to_exclude = self._rows_to_exclude_based_on_qualifier(
            exposure_data, 
            condition_blk_possible_bulk
            )

        # Further filter rows where BLANK_USED is 'N'
        rows_to_exclude = rows_to_exclude & (exposure_data['BLANK_USED'] == 'N')

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_conflicting_qualifier
    def remove_conflicting_qualifier(self, exposure_data):
        '''
        Remove samples with qualifiers conflicting with sample type.
        '''
        exposure_data = exposure_data.copy()

        where_conflict = self._qualif_conv_2020['clean'].isin(['B', 'W'])
        exposure_data = self._remove_based_on_qualifier(
            exposure_data, 
            where_conflict
            )
        return exposure_data
    #endregion

    #region: remove_uninterpretable_qualifier
    def remove_uninterpretable_qualifier(self, exposure_data):
        '''
        Remove samples with qualifiers deemed uninterpretable.
        '''
        exposure_data = exposure_data.copy()

        where_eliminate = self._qualif_conv_2020['clean'] == 'eliminate'
        exposure_data = self._remove_based_on_qualifier(
            exposure_data, 
            where_eliminate
            )
        return exposure_data
    #endregion

    #region: remove_blk_not_bulk
    def remove_blk_not_bulk(self, exposure_data):
        '''
        Remove samples where QUALIFIER is 'BLK' and not possible bulk.
        '''
        exposure_data = exposure_data.copy()

        where_blk_not_bulk  = (
            (self._qualif_conv_2020['clean'] == 'BLK') 
            & (self._qualif_conv_2020['possible_bulk'] == 'N')
        )
        exposure_data = self._remove_based_on_qualifier(
            exposure_data,
            where_blk_not_bulk
            )
        return exposure_data
    #endregion

    #region: _remove_based_on_qualifier
    def _remove_based_on_qualifier(self, exposure_data, condition):
        '''
        General function to remove samples based on QUALIFIER conditions.
        '''
        exposure_data = exposure_data.copy()

        rows_to_exclude = self._rows_to_exclude_based_on_qualifier(
            exposure_data, 
            condition
            )
        return exposure_data.loc[~rows_to_exclude]
    #endregion:

    #region: _rows_to_exclude_based_on_qualifier
    def _rows_to_exclude_based_on_qualifier(
            self,
            exposure_data,
            condition
            ):
        '''
        General function to remove samples based on QUALIFIER conditions.
        '''
        exposure_data = exposure_data.copy()
        
        raw_values_to_exclude = self._qualif_conv_2020.loc[condition, 'raw']
        return exposure_data['QUALIFIER'].isin(raw_values_to_exclude)
    #endregion

    #region: clean_unit_of_measurement
    def clean_unit_of_measurement(self, exposure_data):
        '''
        Clean the `UNIT_OF_MEASUREMENT` column by mapping raw values to clean 
        values.
        '''
        exposure_data = exposure_data.copy()
        
        for clean_value in self._unit_conv_2020['clean'].unique():
            where_clean_value = self._unit_conv_2020['clean'] == clean_value
            raw_values = list(self._unit_conv_2020.loc[where_clean_value, 'raw'])
            where_needs_clean = (
                exposure_data['UNIT_OF_MEASUREMENT'].isin(raw_values)
            )
            exposure_data.loc[where_needs_clean, 'UNIT_OF_MEASUREMENT'] = (
                clean_value
            )

        return exposure_data
    #endregion

    #region: remove_invalid_nondetect
    def remove_invalid_nondetect(self, exposure_data):
        '''
        Remove samples where `QUALIFIER` suggests ND but `SAMPLE_RESULT` > 0
        and not censored (N08), and where `QUALIFIER` suggests ND or is censored
        but `SAMPLE_RESULT` > 0 (N29).
        '''
        exposure_data = exposure_data.copy()

        where_nd = self._qualif_conv_2020['clean'] == 'ND'
        nd_qualifiers = self._qualif_conv_2020.loc[where_nd, 'raw']
        condition_n08 = (
            (exposure_data['SAMPLE_RESULT'] > 0) 
            & (exposure_data['CENSORED'] != 'Y') 
            & (exposure_data['QUALIFIER'].isin(nd_qualifiers))
        )
        exposure_data = exposure_data.loc[~condition_n08]  # N08

        condition_n29 = (
            (exposure_data['SAMPLE_RESULT'] > 0) 
            & ((exposure_data['CENSORED'] == 'Y') 
            | (exposure_data['QUALIFIER'].isin(nd_qualifiers)))
        )
        
        exposure_data = exposure_data.loc[~condition_n29]  # N29

        return exposure_data
    #endregion

    #region: add_censored_column
    def add_censored_column(self, exposure_data):
        '''
        Add a column indicating that the sample is censored ONLY based on the
        'QUALIFIER' column.
        '''
        exposure_data = exposure_data.copy()

        qualifier_censored_values = (
            self.data_settings['qualifier_censored_values']
        )

        exposure_data['CENSORED'] = 'N'  # initialize

        where_censored = (
            exposure_data['QUALIFIER'].isin(qualifier_censored_values)
        )
        exposure_data.loc[where_censored, 'CENSORED'] = 'Y'

        # TODO: Double check whether this is necessary
        # Seems redundant with replace_missing_values()
        exposure_data['QUALIFIER'] = (
            exposure_data['QUALIFIER'].replace('raw was NA', '')
        )

        return exposure_data
    #endregion

    #region: impute_missing_sample_result
    def impute_missing_sample_result(self, exposure_data):
        '''
        Sets missing values (NaN) as zero. 

        Notes
        -----
        This was done in the original R code pipeline and seems to assume that 
        missing values are nondetects.
        '''
        exposure_data = exposure_data.copy()
        exposure_data['SAMPLE_RESULT'] = (
            exposure_data['SAMPLE_RESULT'].fillna(0)
        )
        return exposure_data
    #endregion

    #region: replace_missing_values
    def replace_missing_values(self, exposure_data):
        '''
        '''
        exposure_data = exposure_data.copy()

        for column in ['QUALIFIER', 'UNIT_OF_MEASUREMENT']:
            exposure_data[column] = exposure_data[column].fillna('raw was NA')

        return exposure_data
    #endregion

    #region: remove_limited_sample_substances
    def remove_limited_sample_substances(self, exposure_data):
        '''Exclude substances with few samples.'''
        exposure_data = exposure_data.copy()

        n_for_substance = exposure_data['IMIS_SUBSTANCE_CODE'].value_counts()
        THRES = self.data_settings['min_samples_threshold']
        where_insufficient = n_for_substance < THRES
        limited_substances = list(
            n_for_substance.loc[where_insufficient].keys()
            )

        rows_to_exclude = (
            exposure_data['IMIS_SUBSTANCE_CODE'].isin(limited_substances)
        )
        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_nonchemical_codes
    def remove_nonchemical_codes(self, exposure_data):
        '''Exclude samples with non-chemical IMIS substance codes'''
        exposure_data = exposure_data.copy()

        nonchemical_codes = self.data_settings['nonchemical_codes']
        rows_to_exclude = (
            exposure_data['IMIS_SUBSTANCE_CODE'].isin(nonchemical_codes)
        )

        return exposure_data.loc[~rows_to_exclude]
    #endregion

    #region: remove_nonpersonal
    def remove_nonpersonal(self, exposure_data):
        '''Exclude all samples that are non-personal (e.g., area, etc.)'''
        return super().remove_nonpersonal(exposure_data, 'SAMPLE_TYPE')
    #endregion

    #region: remove_blanks
    def remove_blanks(self, exposure_data):
        '''
        Remove blanks from the 'BLANK_USED' variable 
        
        Other blanks identified later by 'QUALIFIER'.
        '''
        exposure_data = exposure_data.copy()
        not_blank = exposure_data['BLANK_USED'] == 'N'
        return exposure_data.loc[not_blank]
    #endregion

#region: load_unit_measure_conversion
def load_unit_measure_conversion(unit_conv_file):
    '''
    Load conversion table for the 'UNIT_OF_MEASUREMENT' column.
    '''
    unit_conv_2020 = pd.read_csv(unit_conv_file, sep=';')
    return unit_conv_2020
#endregion

#region: load_qualifier_conversion
def load_qualifier_conversion(qualif_conv_file):
    '''
    Load conversion table for the 'QUALIFIER' column.
    '''
    qualif_conv_2020 = pd.read_csv(qualif_conv_file, sep=';')
    return qualif_conv_2020
#endregion