'''
'''

from config_management import config_from_cli_args
import data_management
from pipeline_factory import twostage_estimator_from_config
import results_management
import model_evaluation

if __name__ == '__main__':

    config = config_from_cli_args()

    X_full, y_full = data_management.read_features_and_target(
        config.path['features_file'],
        config.path['target_file'],
        config.data['feature_columns'],
        config.data['log10_features']
    )

    estimator = twostage_estimator_from_config(config.model)
    
    performances = model_evaluation.evaluate_twostage(
        estimator,
        X_full, 
        y_full,
        config
        )
    
    results_management.write_performances(
        performances, 
        config.path['results_dir'], 
        config.file
    )
    results_management.write_metadata(config)