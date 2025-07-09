'''
This module provides functions to persist cross-validation or holdout 
performance dataframes, config settings as metadata, and fitted estimator 
objects to a structured results directory.
'''

import os 
import json 
from joblib import dump

#region: write_performance
def write_performance(performance, results_dir, config_file, filename):
    '''
    Save a performance DataFrame to CSV in a results subdirectory.

    Parameters
    ----------
    performance : pandas.DataFrame
        Performance metrics with MultiIndex columns.
    results_dir : str
        Root directory for all results.
    config_file : str
        Path to the configuration file used to name the subdirectory.
    filename : str
        Name of the CSV file (e.g., 'performances.csv').

    Notes
    -----
    Creates a subdirectory under results_dir named after the config file stem.
    '''
    results_subdir = build_results_subdirectory(results_dir, config_file)
    performance_file = os.path.join(results_subdir, filename)
    performance.to_csv(performance_file)
#endregion

#region: write_metadata
def write_metadata(config):
    '''Write the current configuration settings to JSON.'''
    results_subdir = build_results_subdirectory(
        config.path['results_dir'], 
        config.file
        )
    metadata_file = os.path.join(results_subdir, 'metadata.json')
    with open(metadata_file, 'w') as file:
        json.dump(config.__dict__, file)
#endregion

#region: write_estimator
def write_estimator(estimator, results_dir, config_file):
    '''Serialize and save the fitted estimator using joblib.'''
    results_subdir = build_results_subdirectory(results_dir, config_file)
    estimator_file = os.path.join(results_subdir, 'estimator.joblib')
    dump(estimator, estimator_file)
#endregion

#region: build_results_subdirectory
def build_results_subdirectory(results_dir, config_file):
    '''Construct and ensure existence of a results subdirectory.'''
    stem = os.path.basename(os.path.splitext(config_file)[0])
    results_subdir = os.path.join(results_dir, stem)
    _ensure_directory(results_subdir)
    return results_subdir
#endregion

#region: _ensure_directory
def _ensure_directory(path):
    '''
    Ensure that the specified directory exists.

    If the directory does not exist, it is created.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
#endregion