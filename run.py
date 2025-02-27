'''
'''

import argparse

from config_management import base_cli_parser, UnifiedConfiguration
import data_management
from pipeline_factory import twostage_estimator_from_config
import model_evaluation
import results_management

# TODO: Option to pass a directory of config files and iterate?

#region: parse_cli_args
def parse_cli_args():
    '''
    Parse command-line arguments.

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
    # TODO: Add a shorthand version?
    parser.add_argument(
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

    config = UnifiedConfiguration(
        config_file=args.config_file, 
        encoding=args.encoding
        )
    
    estimator = twostage_estimator_from_config(config.model)

    X_full, y_full = data_management.read_features_and_target(
        config.path['features_file'],
        config.path['target_file'],
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