import pandas as pd

from sklearn.model_selection import GroupKFold

from pandas import testing 

from config_management import config_from_cli_args
import data_management
from metrics_management import metrics_from_config
from model_evaluation import holdout_chemicals, cross_validate_twostage
from pipeline_factory import twostage_estimator_from_config

if __name__ == '__main__':

    config = config_from_cli_args()

    X_full, y_full = data_management.read_features_and_target(
        config.path['features_file'],
        config.path['target_file'],
        config.data['feature_columns'],
        config.data['log10_features']
    )
    chem_groups = y_full.index.get_level_values('DTXSID')
    naics_groups = y_full.index.get_level_values('naics_id')

    # TODO: Evaluate final model on holdout
    y_dev, y_val, dev_mask, val_mask = holdout_chemicals(
        y_full, 
        chem_groups,
        holdout_fraction=config.data['holdout_fraction'],
        random_state=config.data['holdout_random_state']
        )
    X_dev = X_full[dev_mask]
    X_val = X_full[val_mask]
    chem_groups_dev = chem_groups[dev_mask]

    groups_stage2 = None  # by default
    last_step = config.model['stage2'][-1]
    if last_step['class'] == 'MixedLMRegressor':
        # Use 'naics_id' as a grouping variable
        groups_stage2 = naics_groups[dev_mask]

    twostage_estimator = twostage_estimator_from_config(config.model)

    clf_funcs = metrics_from_config(config.metrics['classification'])
    reg_funcs = metrics_from_config(config.metrics['regression'])

    cv = GroupKFold(n_splits=config.data['n_splits_cv'])
    fold_results = cross_validate_twostage(
        twostage_estimator, 
        X_dev, 
        y_dev,
        cv, 
        chem_groups_dev, 
        clf_funcs, 
        reg_funcs,
        groups_stage2=groups_stage2
        )
    
    stored = pd.read_csv(f"{last_step['class']}.csv", index_col=0).squeeze()
    new = pd.DataFrame(fold_results).mean()
    testing.assert_series_equal(new, stored, check_names=False)