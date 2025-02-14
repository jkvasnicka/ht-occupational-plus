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

#region: load_config
def load_config(json_file):
    '''
    '''
    with open(json_file, 'r') as fp:
        return json.load(fp)
#endregion

#region: create_estimator_from_config
def create_estimator_from_config(config):
    '''
    '''
    module = importlib.import_module(config['module'])
    est_class = getattr(module, config['class'])
    params = config.get('parameters', {})
    return est_class(**params)
#endregion

#region: build_metric_func
def build_metric_func(f, kwargs):
    '''
    '''
    return lambda y_true, y_pred: f(y_true, y_pred, **kwargs)
#endregion

#region: load_metric_functions
def load_metric_functions(metric_config):
    '''
    '''
    funcs = {}
    for name, cfg in metric_config.items():
        mod = importlib.import_module(cfg['module'])
        func = getattr(mod, cfg['class'])
        kwargs = cfg.get('kwargs', {})
        funcs[name] = build_metric_func(func, kwargs)
    return funcs
#endregion

###############################################################################
# 2. MixedLMRegressor
###############################################################################

#region: MixedLMRegressor
class MixedLMRegressor(BaseEstimator, RegressorMixin):
    '''
    '''
    #region: fit
    def fit(self, X, y, groups=None):
        '''
        '''
        df, formula = self._prepare_fit_data(X, y)
        if groups is None:
            raise ValueError('MixedLMRegressor requires a "groups" parameter.')
        groups = np.asarray(groups)
        self.model_ = smf.mixedlm(formula=formula, data=df,
                                  groups=groups, re_formula='1')
        self.result_ = self.model_.fit(method='bfgs', reml=False, maxiter=200)
        return self
    #endregion
    
    #region: predict
    def predict(self, X):
        '''
        '''
        X_df = self._ensure_dataframe(np.asarray(X))
        return self.result_.predict(X_df)
    #endregion
    
    #region: get_icc
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
    #endregion

    #region: _ensure_dataframe
    def _ensure_dataframe(self, X):
        '''
        '''
        if not isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
            cols = [f'x{i}' for i in range(n_features)]
            X = pd.DataFrame(X, columns=cols)
        return X
    #endregion
    
    #region: _prepare_fit_data    
    def _prepare_fit_data(self, X, y):
        '''
        '''
        X_df = self._ensure_dataframe(np.asarray(X))
        df = X_df.copy()
        df['y'] = np.asarray(y).ravel()
        predictors = [col for col in df.columns if col != 'y']
        formula = 'y ~ 1'
        if predictors:
            formula += ' + ' + ' + '.join(predictors)
        return df, formula
    #endregion
#endregion

###############################################################################
# 3. TwoStageEstimator
###############################################################################

#region: TwoStageEstimator
class TwoStageEstimator(BaseEstimator):
    '''
    '''
    #region: __init__
    def __init__(self, stage1_estimator, stage2_estimator,
                 target_transform=np.log10,
                 target_inverse_transform=lambda x: 10 ** x):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.target_transform = target_transform
        self.target_inverse_transform = target_inverse_transform
    #endregion

    #region: fit
    def fit(self, X, y, groups_stage2=None):
        '''
        '''
        self._fit_stage1(X, y)
        self._fit_stage2(X, y, groups_stage2=groups_stage2)
        return self
    #endregion

    #region: _fit_stage1
    def _fit_stage1(self, X, y):
        '''
        '''
        y = np.asarray(y).ravel()
        detection = self._get_detected_mask(y).astype(int)
        self.stage1_estimator.fit(X, detection)
    #endregion
    
    #region: _fit_stage2
    def _fit_stage2(self, X, y, groups_stage2=None):
        '''
        '''
        y = np.asarray(y).ravel()
        mask = self._get_detected_mask(y)
        if mask.sum() == 0:
            raise ValueError('No detected samples to fit stage-2.')
        X_det = np.asarray(X)[mask]
        y_det = y[mask]
        y_det_trans = self.target_transform(y_det)
        final_est = self.stage2_estimator
        if hasattr(self.stage2_estimator, 'steps'):
            final_est = self.stage2_estimator.steps[-1][1]
        fit_params = {}
        if (groups_stage2 is not None and 
                isinstance(final_est, MixedLMRegressor)):
            groups_arr = np.asarray(groups_stage2)
            groups_det = groups_arr[mask]
            if hasattr(self.stage2_estimator, 'steps'):
                fit_params = {'regressor__groups': groups_det}
            else:
                fit_params = {'groups': groups_det}
        self.stage2_estimator.fit(X_det, y_det_trans, **fit_params)
    #endregion
         
    #region: predict
    def predict(self, X):
        '''
        '''
        X = np.asarray(X)
        detect_pred = self._predict_stage1(X)
        y_pred = np.zeros(X.shape[0])
        idx = np.where(detect_pred == 1)[0]
        if idx.size > 0:
            X_det = X[idx]
            y_pred_trans = self._predict_stage2(X_det)
            y_pred[idx] = self.target_inverse_transform(y_pred_trans)
        return y_pred
    #endregion

    #region: _predict_stage1
    def _predict_stage1(self, X):
        '''
        '''
        X = np.asarray(X)
        return self.stage1_estimator.predict(X)
    #endregion
    
    #region: _predict_stage2
    def _predict_stage2(self, X):
        '''
        '''
        X = np.asarray(X)
        return self.stage2_estimator.predict(X)
    #endregion

    #region: _get_detected_mask
    def _get_detected_mask(self, y):
        '''
        '''
        return (y > 0)
    #endregion
#endregion

###############################################################################
# 4. Cross-Validation Helpers
###############################################################################

#region: cross_validate_twostage
def cross_validate_twostage(
        estimator, 
        X, 
        y, 
        cv, 
        groups_cv, 
        clf_funcs, 
        reg_funcs, 
        groups_stage2=None
        ):
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
        metrics = _evaluate_fold(estimator, X_test, y_test,
                                 clf_funcs, reg_funcs,
                                 estimator.target_transform)
        fold_results.append(metrics)
    return fold_results
#endregion

#region: _evaluate_fold
def _evaluate_fold(estimator, X_test, y_test, clf_funcs, reg_funcs,
                   target_transform):
    '''
    '''
    y_pred = estimator.predict(X_test)
    # Attempt to obtain probability estimates for AUC.
    try:
        y_proba = estimator.stage1_estimator.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None
    clf_metrics = _classification_metrics(
        (y_test > 0).astype(int),
        (y_pred > 0).astype(int),
        clf_funcs,
        y_proba=y_proba
    )
    reg_metrics = _regression_metrics(
        y_test, 
        y_pred, 
        reg_funcs,
        target_transform
        )
    fold_metrics = {}
    fold_metrics.update(clf_metrics)
    fold_metrics.update(reg_metrics)
    return fold_metrics
#endregion

#region: _classification_metrics
def _classification_metrics(y_true, y_pred, metric_funcs, y_proba=None):
    '''
    '''
    metrics = {}
    for key, func in metric_funcs.items():
        try:
            if key == 'auc' and y_proba is not None:
                metrics[key] = func(y_true, y_proba)
            else:
                metrics[key] = func(y_true, y_pred)
        except Exception:
            metrics[key] = np.nan
    return metrics
#endregion

#region: _regression_metrics
def _regression_metrics(y_true, y_pred, metric_funcs, target_transform):
    '''
    '''
    mask = (y_true > 0) & (y_pred > 0)
    metrics = {}
    if np.sum(mask) > 0:
        y_true_trans = target_transform(y_true[mask])
        y_pred_trans = target_transform(y_pred[mask])
        for key, func in metric_funcs.items():
            try:
                metrics[key] = func(y_true_trans, y_pred_trans)
            except Exception:
                metrics[key] = np.nan
    else:
        for key in metric_funcs.keys():
            metrics[key] = np.nan
    return metrics
#endregion

#region: holdout_chemicals
def holdout_chemicals(y, chem_group, holdout_fraction=0.1,
                        random_state=42):
    '''Create hold-out sets by chemical.'''
    unique = np.unique(chem_group)
    n_holdout = int(len(unique) * holdout_fraction)
    rng = np.random.RandomState(random_state)
    holdout = rng.choice(unique, size=n_holdout, replace=False)
    mask = np.isin(chem_group, holdout)
    return y[~mask], y[mask], ~mask, mask
#endregion
 
###############################################################################
# 5. Data Loading and Preparation Helper
###############################################################################

#region: load_and_prepare_data
def load_and_prepare_data():
    '''
    '''
    config = UnifiedConfiguration()
    ec = data_management.read_targets(config.path['target_dir'])
    sorted_levels = config.cehd['naics_levels']
    ec = {k: ec[k] for k in sorted_levels}
    y_df = ec['sector'].copy()
    feat = pd.read_parquet(config.path['opera_features_file'])
    props = ['VP_pred', 'KOA_pred', 'MolWeight', 'TopoPolSurfAir']
    X_df, y_df = feat[props].align(y_df, join='inner', axis=0)
    # Upstream log10 transform of selected features
    X_df['VP_pred'] = np.log10(X_df['VP_pred'])
    X_df['KOA_pred'] = np.log10(X_df['KOA_pred'])
    X_df['MolWeight'] = np.log10(X_df['MolWeight'])
    groups_cv = y_df.index.get_level_values('DTXSID').to_numpy()
    groups_stage2 = y_df.index.get_level_values('naics_id').to_numpy()
    return X_df.to_numpy(), y_df.to_numpy().ravel(), groups_cv, groups_stage2
#endregion

###############################################################################
# 6. Main Execution
###############################################################################

#region: __main__
if __name__ == '__main__':

    # Example metric configurations.
    metric_config_classification = {
        'accuracy': {'module': 'sklearn.metrics',
                    'class': 'accuracy_score',
                    'kwargs': {}},
        'precision': {'module': 'sklearn.metrics',
                    'class': 'precision_score',
                    'kwargs': {'zero_division': 0}},
        'recall': {'module': 'sklearn.metrics',
                'class': 'recall_score',
                'kwargs': {'zero_division': 0}},
        'f1': {'module': 'sklearn.metrics',
            'class': 'f1_score',
            'kwargs': {'zero_division': 0}},
        'auc': {'module': 'sklearn.metrics',
                'class': 'roc_auc_score',
                'kwargs': {}}
    }

    metric_config_regression = {
        'r2_score': {'module': 'sklearn.metrics',
                    'class': 'r2_score',
                    'kwargs': {}},
        'root_mean_squared_error': {'module': 'sklearn.metrics',
                                    'class': 'mean_squared_error',
                                    'kwargs': {'squared': False}},
        'median_absolute_error': {'module': 'sklearn.metrics',
                                'class': 'median_absolute_error',
                                'kwargs': {}}
    }
    clf_funcs = load_metric_functions(metric_config_classification)
    reg_funcs = load_metric_functions(metric_config_regression)

    X_full, y_full, chem_group, naics_group = load_and_prepare_data()

    y_dev, y_val, dev_mask, val_mask = holdout_chemicals(y_full, chem_group)
    X_dev = X_full[dev_mask]
    X_val = X_full[val_mask]
    chem_group_dev = chem_group[dev_mask]
    naics_group_dev = naics_group[dev_mask]

    # Define pipelines for Stage 1 and Stage 2.
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    stage1_pipe = Pipeline([
        ('preprocessor', preproc),
        ('classifier', LogisticRegression())
    ])

    stage2_pipe_ols = Pipeline([
        ('preprocessor', preproc),
        ('regressor', LinearRegression())
    ])
    two_stage_ols = TwoStageEstimator(
        stage1_estimator=stage1_pipe,
        stage2_estimator=stage2_pipe_ols
        )

    stage2_pipe_mixed = Pipeline([
        ('preprocessor', preproc),
        ('regressor', MixedLMRegressor())
    ])
    two_stage_mixed = TwoStageEstimator(
        stage1_estimator=stage1_pipe,
        stage2_estimator=stage2_pipe_mixed
        )

    cv = GroupKFold(n_splits=5)
    results_ols = cross_validate_twostage(
        two_stage_ols, 
        X_dev, 
        y_dev,
        cv, 
        chem_group_dev, 
        clf_funcs, 
        reg_funcs,
        groups_stage2=None
        )
    results_mixed = cross_validate_twostage(
        two_stage_mixed, 
        X_dev, 
        y_dev,
        cv, 
        chem_group_dev,
        clf_funcs, 
        reg_funcs,
        groups_stage2=naics_group_dev
        )

    stored_ols = pd.read_csv('results_ols.csv', index_col=0).squeeze()
    stored_mixed = pd.read_csv('results_mixed.csv', index_col=0).squeeze()

    new_ols = pd.DataFrame(results_ols).mean()
    new_mixed = pd.DataFrame(results_mixed).mean()

    testing.assert_series_equal(new_ols, stored_ols, check_names=False)
    testing.assert_series_equal(new_mixed, stored_mixed, check_names=False)
#endregion