'''
Main entry point for the two-stage modeling workflow.

Runs cross-validation or final holdout evaluation for one or more two-stage
estimator instances defined via config files. Persists results, fitted 
estimator, and config settings as metadata. 

The user must specify either a path to a single 'main' config file 
(--config_file) for a single estimator, or a path to a directory of multiple 
'main' config files (--config_dir). The latter is used for sensitivity 
analyses (e.g., OLS vs. MixedLM regressors).

Parameters
-----------
-c, --config_file : str, optional
    Path to a single config file.
-d, --config_dir : str, optional
    Path to a directory of config files for sensitivity analyses.
-e, --encoding : str, optional
    Encoding for the configuration files (default: 'utf-8').
-t, --evaluation_type : {'cv', 'holdout'}, default: 'cv'
    Evaluation mode: 
      - 'cv' : perform k-fold cross-validation on the development set,
               grouped by chemical 
      - 'holdout' : fit on development set and evaluate on holdout set  

Notes
-----
Cross validation should be used for model selection, whereas holdout evaluation
is then used to assess the selected model's generalization to the "unseen"
holdout set.

Examples
--------
# 1. Cross-validate several two-stage estimators for model selection:
$ python run.py -d config_main

# 2. Evaluate holdout performance for a selected estimator:
$ python run.py -c config_main/config_ols.json -t holdout
'''

import argparse
import os

from config_management import base_cli_parser, UnifiedConfiguration
import data_management
from pipeline_factory import twostage_estimator_from_config
import model_evaluation
import results_management

#region: parse_cli_args
def parse_cli_args():
    '''
    Parse command-line arguments for configuring and executing model
    evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    '''
    parent = base_cli_parser()
    parser = argparse.ArgumentParser(
        parents=[parent],
        description='Run model fitting, evaluation, & results management.'
    )
    parser.add_argument(
        '-d',
        '--config_dir',
        type=str, 
        help='Path to a directory of multiple main configuration files'
    )
    parser.add_argument(
        '-t',
        '--evaluation_type',
        type=str,
        choices=['cv', 'holdout'],
        default='cv',
        help='Type of evaluation to perform: "cv" for cross-validation, '
             '"holdout" for final holdout evaluation'
    )
    return parser.parse_args()
#endregion

if __name__ == '__main__':

    args = parse_cli_args()

    if args.config_file:
        config_files = [args.config_file]
    else:
        config_files = [
            os.path.join(args.config_dir, f)
            for f in os.listdir(args.config_dir)
            if f.endswith('.json')
        ]

    for config_file in config_files:

        config = UnifiedConfiguration(
            config_file=config_file, 
            encoding=args.encoding
            )
        
        estimator = twostage_estimator_from_config(config.model)

        X_full, y_full = data_management.prepare_features_and_target(
            config.usis, 
            config.cehd,
            config.path, 
            config.comptox,
            config.data['feature_columns'],
            config.data['log10_features']
            )

        model_evaluation.evaluate_twostage(
            estimator,
            X_full, 
            y_full,
            config,
            args.evaluation_type
        )
            
        results_management.write_metadata(config)