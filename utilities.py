'''
'''
import pandas as pd 
import numpy as np
import os
from functools import reduce
import re
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from rdkit.Chem import PandasTools
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, classification_report

def load_cdr_data(csv_file, data_config):
    '''
    '''
    chem_id_type_col = data_config['chem_id_type_col']

    data = pd.read_csv(csv_file).dropna(how='all')
    # Filter out chemicals with no CASRN
    data = data.loc[data[chem_id_type_col] == 'CASRN']

    if 'index_col' in data_config:
        data = data.set_index(data_config['index_col'])
        
    return data

def load_qsar_ready_smiles(smi_file):
    '''
    '''
    smiles_for_casrn = {}  # initialize
    with open(smi_file, 'r') as smi:
        for line in smi:
            smiles, casrn =line.strip().split('\t')
            smiles_for_casrn[casrn] = smiles
    return smiles_for_casrn

def extract_physical_forms(forms_series):
    '''
    '''
    return forms_series.apply(physical_forms_from_value)

def physical_forms_from_value(value):
    if pd.isna(value) or value == 'CBI':
        return []
    else:
        return [value.strip() for value in value.split(',')]

def extract_subsectors(naics_data):
    '''
    '''
    combined_subsectors = (
        naics_data
        .apply(extract_naics_codes, axis=1)
        .apply(subsectors_from_naics)
    )
    return combined_subsectors

def extract_sectors(combined_subsectors):
    '''
    '''
    return combined_subsectors.apply(sectors_from_subsectors)

def extract_naics(code):
    if pd.isna(code) or code == 'CBI':
        return None
    # Use a regex that captures any sequence of exactly six digits
    match = re.findall(r'\d{6}', code)
    return match[0] if match else None

def extract_naics_codes(row):
    # Use list comprehension to apply extract_naics to each item in the row
    codes = [extract_naics(naics) for naics in row]
    # Filter out None values and return a unique set of results
    return list(set(naics for naics in codes if naics))

def extract_subsector(naics):
    return naics[:3]

def extract_sector(naics):
    return naics[:2]

def subsectors_from_naics(naics_codes):
    return [extract_subsector(naics) for naics in naics_codes]

def sectors_from_subsectors(subsectors):
    return list(set(extract_sector(sub) for sub in subsectors))

def create_binary_data(category_series, by=None, agg_func='max', drop_columns=None):

    flattened_categories = [
        item for sublist in category_series 
        for item in sublist if item
    ]
    unique_categories = list(set(flattened_categories))

    new_columns = {}  # initialize
    
    # Generate binary columns and store in dictionary
    for category in unique_categories:
        new_columns[category] = category_series.apply(lambda categories: 1 if category in categories else 0)
    
    binary_data = pd.DataFrame(new_columns, index=category_series.index)
    
    binary_data = aggregate_data(binary_data, by=by, agg_func=agg_func)

    if drop_columns:
        binary_data = binary_data.drop(drop_columns, axis=1)

    return binary_data

def create_continuous_data(string_data, by=None, kwargs_for_col=None):
    '''
    '''
    continuous_data = {}  # initialize
    for col, string_series in string_data.items():
        kwargs = kwargs_for_col[col]
        continuous_data[col] = string_series.apply(
            string_to_numerical, 
            **kwargs
            )    
    continuous_data = pd.DataFrame(continuous_data, index=string_data.index)

    return aggregate_data(continuous_data, by=by, **kwargs)

def string_to_numerical(value, lower_bound=None, upper_bound=None, **kwargs):
    
    if pd.isna(value) or value == 'CBI' or 'Not Known' in value:
        return np.nan
        
    value = value.replace(',', '').replace('%', '').replace(' ', '')

    if '–' in value:
        # Convert range to midpoint
        bounds = value.split('–')
        lb = float(bounds[0])
        ub = float(bounds[-1].replace('<', ''))
        value = (lb + ub) / 2
    elif '<' in value:
        value = float(value.replace('<', ''))
        value = (lower_bound + value) / 2
    elif '+' in value:
        value = float(value.replace('+', ''))
        value = (value + upper_bound) / 2
    
    return float(value)

def aggregate_data(data, by=None, agg_func='max', **kwargs):
    '''
    '''
    # Aggregate the data by chemical ID. Each row is a unique chemical.
    groupby = data.reset_index().groupby(by=by)
    return getattr(groupby, agg_func)()

def with_common_index(X, y):
    common_chem_ids = X.index.intersection(y.index)
    X = X.loc[common_chem_ids]
    y = y.loc[common_chem_ids]
    return X, y