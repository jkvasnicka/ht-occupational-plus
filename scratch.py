import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Seed for reproducibility
np.random.seed(42)

from pandas import testing 

from config_management import UnifiedConfiguration
import data_management
from metrics_management import metrics_from_config
from model_evaluation import holdout_chemicals, cross_validate_twostage

from mixedlm_estimator import MixedLMRegressor
from twostage_estimator import TwoStageEstimator

config = UnifiedConfiguration()

#region: __main__
if __name__ == '__main__':

    X_full, y_full = data_management.read_features_and_target(
        config.path['features_file'],
        config.path['target_file'],
        config.model['feature_columns'],
        config.model['log10_features']
    )
    chem_groups = y_full.index.get_level_values('DTXSID')
    naics_groups = y_full.index.get_level_values('naics_id')

    y_dev, y_val, dev_mask, val_mask = holdout_chemicals(y_full, chem_groups)
    X_dev = X_full[dev_mask]
    X_val = X_full[val_mask]
    chem_groups_dev = chem_groups[dev_mask]
    naics_groups_dev = naics_groups[dev_mask]

    # TODO: Consider a pipeline builder. Move params to config
    # Define pipelines for Stage 1 and Stage 2.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    stage1_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    stage2_pipe_ols = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    two_stage_ols = TwoStageEstimator(
        stage1_estimator=stage1_pipe,
        stage2_estimator=stage2_pipe_ols
        )

    stage2_pipe_mixed = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MixedLMRegressor())
    ])
    two_stage_mixed = TwoStageEstimator(
        stage1_estimator=stage1_pipe,
        stage2_estimator=stage2_pipe_mixed
        )

    clf_funcs = metrics_from_config(config.metrics['classification'])
    reg_funcs = metrics_from_config(config.metrics['regression'])

    # TODO: Set a random seed properly
    cv = GroupKFold(n_splits=5)
    results_ols = cross_validate_twostage(
        two_stage_ols, 
        X_dev, 
        y_dev,
        cv, 
        chem_groups_dev, 
        clf_funcs, 
        reg_funcs,
        groups_stage2=None
        )
    results_mixed = cross_validate_twostage(
        two_stage_mixed, 
        X_dev, 
        y_dev,
        cv, 
        chem_groups_dev,
        clf_funcs, 
        reg_funcs,
        groups_stage2=naics_groups_dev
        )

    stored_ols = pd.read_csv('results_ols.csv', index_col=0).squeeze()
    stored_mixed = pd.read_csv('results_mixed.csv', index_col=0).squeeze()

    new_ols = pd.DataFrame(results_ols).mean()
    new_mixed = pd.DataFrame(results_mixed).mean()

    testing.assert_series_equal(new_ols, stored_ols, check_names=False)
    testing.assert_series_equal(new_mixed, stored_mixed, check_names=False)
#endregion