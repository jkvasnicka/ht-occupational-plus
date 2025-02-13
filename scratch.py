import json
import importlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, r2_score, mean_squared_error, median_absolute_error)
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Seed for reproducibility
np.random.seed(42)

from pandas import testing 

from config_management import UnifiedConfiguration
import data_management

config = UnifiedConfiguration()

###############################################################################
# 1. Helper Functions and Metric Loader
###############################################################################

def load_config(json_file):
    '''
    '''
    with open(json_file, 'r') as fp:
        config = json.load(fp)
    return config

def create_estimator_from_config(config):
    '''
    '''
    module = importlib.import_module(config['module'])
    estimator_class = getattr(module, config['class'])
    params = config.get('parameters', {})
    return estimator_class(**params)

def load_metric_functions(metric_config):
    '''
    '''
    funcs = {}
    for name, cfg in metric_config.items():
        mod = importlib.import_module(cfg['module'])
        func = getattr(mod, cfg['class'])
        kwargs = cfg.get('kwargs', {})
        def make_func(f, kw):
            return lambda y_true, y_pred: f(y_true, y_pred, **kw)
        funcs[name] = make_func(func, kwargs)
    return funcs

# Example metric configuration dictionaries
metric_config_classification = {
    'accuracy': {'module': 'sklearn.metrics', 'class': 'accuracy_score', 'kwargs': {}},
    'precision': {'module': 'sklearn.metrics', 'class': 'precision_score', 'kwargs': {'zero_division': 0}},
    'recall': {'module': 'sklearn.metrics', 'class': 'recall_score', 'kwargs': {'zero_division': 0}},
    'f1': {'module': 'sklearn.metrics', 'class': 'f1_score', 'kwargs': {'zero_division': 0}},
    'auc': {'module': 'sklearn.metrics', 'class': 'roc_auc_score', 'kwargs': {}}
}

metric_config_regression = {
    'r2_score': {'module': 'sklearn.metrics', 'class': 'r2_score', 'kwargs': {}},
    'root_mean_squared_error': {'module': 'sklearn.metrics', 'class': 'mean_squared_error', 'kwargs': {'squared': False}},
    'median_absolute_error': {'module': 'sklearn.metrics', 'class': 'median_absolute_error', 'kwargs': {}}
}

clf_metrics_funcs = load_metric_functions(metric_config_classification)
reg_metrics_funcs = load_metric_functions(metric_config_regression)

###############################################################################
# 2. MixedLMRegressor
###############################################################################

class MixedLMRegressor(BaseEstimator, RegressorMixin):
    '''
    '''
    def _ensure_dataframe(self, X):
        '''
        '''
        if not isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
            cols = [f'x{i}' for i in range(n_features)]
            X = pd.DataFrame(X, columns=cols)
        return X

    def fit(self, X, y, groups=None):
        '''
        '''
        X_df = self._ensure_dataframe(np.asarray(X))
        df = X_df.copy()
        df['y'] = np.asarray(y).ravel()
        if groups is None:
            raise ValueError('MixedLMRegressor requires a "groups" parameter for the random intercept.')
        groups = np.asarray(groups)
        predictors = [col for col in df.columns if col != 'y']
        formula = 'y ~ 1'
        if predictors:
            formula += ' + ' + ' + '.join(predictors)
        self.model_ = smf.mixedlm(formula=formula, data=df, groups=groups, re_formula='1')
        self.result_ = self.model_.fit(method='bfgs', reml=False, maxiter=200)
        return self

    def predict(self, X):
        '''
        '''
        X_df = self._ensure_dataframe(np.asarray(X))
        return self.result_.predict(X_df)

    def get_icc(self):
        '''
        '''
        check_is_fitted(self, 'result_')
        if self.result_.cov_re.shape[0] > 0:
            var_group = self.result_.cov_re.iloc[0, 0]
        else:
            var_group = 0.0
        var_resid = self.result_.scale
        return var_group / (var_group + var_resid)

###############################################################################
# 3. TwoStageEstimator
###############################################################################

class TwoStageEstimator(BaseEstimator):
    '''
    '''
    def __init__(self, stage1_estimator, stage2_estimator,
                 target_transform=np.log10,
                 target_inverse_transform=lambda x: 10 ** x):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.target_transform = target_transform
        self.target_inverse_transform = target_inverse_transform

    def fit(self, X, y, groups_stage2=None):
        '''
        '''
        y = np.asarray(y).ravel()
        self.detection_ = (y > 0).astype(int)
        self.stage1_estimator.fit(X, self.detection_)
        mask = self.detection_ == 1
        if mask.sum() == 0:
            raise ValueError('No detected samples in training data; cannot fit stage-2.')
        X_detect = np.asarray(X)[mask]
        y_detect = y[mask]
        y_detect_trans = self.target_transform(y_detect)
        # Determine if stage2_estimator requires groups by checking its final estimator
        final_est = self.stage2_estimator
        if hasattr(self.stage2_estimator, 'steps'):
            final_est = self.stage2_estimator.steps[-1][1]
        fit_params = {}
        if groups_stage2 is not None and isinstance(final_est, MixedLMRegressor):
            groups_arr = np.asarray(groups_stage2)
            groups_for_detect = groups_arr[mask]
            # If stage2_estimator is a Pipeline with final step 'regressor', use key 'regressor__groups'
            if hasattr(self.stage2_estimator, 'steps'):
                fit_params = {'regressor__groups': groups_for_detect}
            else:
                fit_params = {'groups': groups_for_detect}
        self.stage2_estimator.fit(X_detect, y_detect_trans, **fit_params)
        return self

    def predict(self, X):
        '''
        '''
        X = np.asarray(X)
        detect_pred = self.stage1_estimator.predict(X)
        y_pred = np.zeros(X.shape[0])
        if np.sum(detect_pred) > 0:
            idx = np.where(detect_pred == 1)[0]
            X_detect = X[idx]
            stage2_pred_trans = self.stage2_estimator.predict(X_detect)
            stage2_pred = self.target_inverse_transform(stage2_pred_trans)
            y_pred[idx] = stage2_pred
        return y_pred

###############################################################################
# 4. Cross-Validation Function
###############################################################################

def cross_validate_twostage(estimator, X, y, cv, groups_cv, groups_stage2=None,
                            clf_metric_funcs=clf_metrics_funcs,
                            reg_metric_funcs=reg_metrics_funcs):
    '''
    '''
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    fold_results = []
    for train_idx, test_idx in cv.split(X, y, groups=groups_cv):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if groups_stage2 is not None:
            groups_stage2_train = np.asarray(groups_stage2)[train_idx]
        else:
            groups_stage2_train = None

        estimator.fit(X_train, y_train, groups_stage2=groups_stage2_train)
        y_pred = estimator.predict(X_test)
        detect_true = (y_test > 0).astype(int)
        detect_pred = (y_pred > 0).astype(int)

        fold_metrics = {}
        # Classification metrics (applied on full test set)
        for key, func in clf_metric_funcs.items():
            try:
                if key == 'auc':
                    try:
                        y_proba = estimator.stage1_estimator.predict_proba(X_test)[:, 1]
                        fold_metrics[key] = func(detect_true, y_proba)
                    except Exception:
                        fold_metrics[key] = np.nan
                else:
                    fold_metrics[key] = func(detect_true, detect_pred)
            except Exception:
                fold_metrics[key] = np.nan

        # Regression metrics only for samples where both true and predicted are > 0
        mask = (y_test > 0) & (y_pred > 0)
        if mask.sum() > 0:
            y_test_trans = estimator.target_transform(y_test[mask])
            y_pred_trans = estimator.target_transform(y_pred[mask])
            for key, func in reg_metric_funcs.items():
                try:
                    fold_metrics[key] = func(y_test_trans, y_pred_trans)
                except Exception:
                    fold_metrics[key] = np.nan
        else:
            for key in reg_metric_funcs.keys():
                fold_metrics[key] = np.nan

        fold_results.append(fold_metrics)
    return fold_results

###############################################################################
# 5. Example Usage with Real Data
###############################################################################

if __name__ == '__main__':
    # Load targets
    ec_for_naics = data_management.read_targets(config.path['target_dir'])
    sorted_naics_levels = config.cehd['naics_levels']
    ec_for_naics = {k: ec_for_naics[k] for k in sorted_naics_levels}
    # Choose the target series
    y_full_df = ec_for_naics['sector'].copy()

    # Load features
    opera_features = pd.read_parquet(config.path['opera_features_file'])
    property_columns = ['VP_pred', 'KOA_pred', 'MolWeight', 'TopoPolSurfAir']
    X_full_df, y_full_df = opera_features[property_columns].align(y_full_df, join='inner', axis=0)

    X_full_df['VP_pred'] = np.log10(X_full_df['VP_pred'])
    X_full_df['KOA_pred'] = np.log10(X_full_df['KOA_pred'])
    X_full_df['MolWeight'] = np.log10(X_full_df['MolWeight'])

    # Extract grouping variables from the MultiIndex
    chem_group = y_full_df.index.get_level_values('DTXSID').to_numpy()  # for CV splitting
    naics_group = y_full_df.index.get_level_values('naics_id').to_numpy()  # for MixedLM

    # Convert to NumPy arrays for scikit-learn
    X_full = X_full_df.to_numpy()
    y_full = y_full_df.to_numpy().ravel()

    # Create hold-out sets by chemical
    def holdout_chemicals(y, chem_group, holdout_fraction=0.1, random_state=42):
        unique_chems = np.unique(chem_group)
        n_holdout = int(len(unique_chems) * holdout_fraction)
        rng = np.random.RandomState(random_state)
        holdout = rng.choice(unique_chems, size=n_holdout, replace=False)
        mask = np.isin(chem_group, holdout)
        return y[~mask], y[mask], ~mask, mask

    y_dev, y_val, dev_mask, val_mask = holdout_chemicals(y_full, chem_group)
    X_dev = X_full[dev_mask]
    X_val = X_full[val_mask]
    chem_group_dev = chem_group[dev_mask]
    naics_group_dev = naics_group[dev_mask]

    ###############################################################################
    # Define Stage-1 and Stage-2 Estimators via Pipelines.
    ###############################################################################

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Stage 1: Classification pipeline
    stage1_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Option A: Stage 2 using OLS
    stage2_pipeline_ols = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    two_stage_ols = TwoStageEstimator(stage1_estimator=stage1_pipeline,
                                    stage2_estimator=stage2_pipeline_ols)

    # Option B: Stage 2 using Mixed Effects
    stage2_pipeline_mixed = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MixedLMRegressor())
    ])
    two_stage_mixed = TwoStageEstimator(stage1_estimator=stage1_pipeline,
                                        stage2_estimator=stage2_pipeline_mixed)

    ###############################################################################
    # Cross-Validation
    ###############################################################################
    cv = GroupKFold(n_splits=5)
    # For CV splitting, use chem_group_dev; for stage2 groups, use naics_group_dev
    results_ols = cross_validate_twostage(two_stage_ols, X_dev, y_dev, cv=cv,
                                        groups_cv=chem_group_dev,
                                        groups_stage2=None)  # OLS ignores groups
    results_mixed = cross_validate_twostage(two_stage_mixed, X_dev, y_dev, cv=cv,
                                            groups_cv=chem_group_dev,
                                            groups_stage2=naics_group_dev)

    stored_ols = pd.read_csv('results_ols.csv', index_col=0).squeeze()
    stored_mixed = pd.read_csv('results_mixed.csv', index_col=0).squeeze()

    new_ols = pd.DataFrame(results_ols).mean()
    new_mixed = pd.DataFrame(results_mixed).mean()

    testing.assert_series_equal(new_ols, stored_ols, check_names=False)
    testing.assert_series_equal(new_mixed, stored_mixed, check_names=False)