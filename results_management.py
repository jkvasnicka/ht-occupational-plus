'''
'''

import os 
import json 
import pandas as pd

#region: write_performances
def write_performances(performances, results_dir, config_file):
    '''
    '''
    results_subdir = build_results_subdirectory(results_dir, config_file)
    performances_file = os.path.join(results_subdir, 'performances.csv')
    performances.to_csv(performances_file)
#endregion

#region: read_performances
def read_performances(performances_file):
    '''
    '''
    return pd.read_csv(performances_file, index_col=0, header=[0, 1])
#endregion

#region: write_metadata
def write_metadata(config):
    '''
    '''
    results_subdir = build_results_subdirectory(
        config.path['results_dir'], 
        config.file
        )
    metadata_file = os.path.join(results_subdir, 'metadata.json')
    with open(metadata_file, 'w') as file:
        json.dump(config.__dict__, file)
#endregion

#region: read_metadata
def read_metadata(metadata_file):
    '''
    '''
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
    return metadata
#endregion

#region: build_results_subdirectory
def build_results_subdirectory(results_dir, config_file):
    '''
    '''
    stem = os.path.basename(os.path.splitext(config_file)[0])
    results_subdir = os.path.join(results_dir, stem)
    _ensure_directory(results_subdir)
    return results_subdir
#endregion

#region: _ensure_directory
def _ensure_directory(path):
    '''
    Ensure that the specified directory exists.

    Parameters
    ----------
    path : str
        Path to the directory.

    Notes
    -----
    If the directory does not exist, it is created.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
#endregion