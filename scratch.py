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

#region: create_estimator_from_config
def create_estimator_from_config(config):
    '''
    '''
    module = importlib.import_module(config['module'])
    est_class = getattr(module, config['class'])
    params = config.get('parameters', {})
    return est_class(**params)
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

#region: build_metric_func
def build_metric_func(f, kwargs):
    '''
    '''
    return lambda y_true, y_pred: f(y_true, y_pred, **kwargs)
#endregion

###############################################################################
# 2. MixedLMRegressor
###############################################################################

#region: MixedLMRegressor
class MixedLMRegressor(BaseEstimator, RegressorMixin):
    '''
    '''
    def __init__(self, method='bfgs', reml=False, maxiter=200):
        self.method = method
        self.reml = reml
        self.maxiter = maxiter

    #region: fit
    def fit(self, X, y, groups=None):
        '''
        '''
        X = np.asarray(X)
        y = np.asarray(y)

        Xy_df, formula = self._prepare_fit_data(X, y)

        if groups is None:
            raise ValueError('MixedLMRegressor requires a "groups" parameter.')
        groups = np.asarray(groups)

        self.model_ = smf.mixedlm(
            formula=formula, 
            data=Xy_df,
            groups=groups,
            re_formula='1'  # random intercept–only model
            )
        self.result_ = self.model_.fit(
            method=self.method, 
            reml=self.reml, 
            maxiter=self.maxiter
            )

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
        
        return intraclass_correlation(var_group, var_resid)
    #endregion

    #region: _ensure_dataframe
    def _ensure_dataframe(self, X):
        '''
        Convert X to a DataFrame with generic column names.

        This is necessary because statsmodels' formula interface requires a 
        DataFrame with valid column names. Using integer column names (the 
        default) may not produce a usable formula string.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        DataFrame
            DataFrame version of X with columns named 'x0', 'x1', ...
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
        X_df = self._ensure_dataframe(X)
        Xy_df = X_df.copy()
        Xy_df['y'] = y

        predictors = [col for col in Xy_df.columns if col != 'y']
        formula = 'y ~ 1'  # Start with intercept-only model
        if predictors:
            formula += ' + ' + ' + '.join(predictors)

        return Xy_df, formula
    #endregion
#endregion

# TODO: Make static method of MixedLMRegressor?
#region: intraclass_correlation
def intraclass_correlation(var_group, var_resid):
    '''
    Computes Intraclass Correlation Coefficient (ICC).
    
    Parameters
    ----------
    var_group :
        Random intercept variance (mdf.cov_re.iloc[0,0]).
    var_resid : 
        Residual variance (mdf.scale).
    '''
    return var_group / (var_group + var_resid)
#endregion

###############################################################################
# 3. TwoStageEstimator
###############################################################################

#region: TwoStageEstimator
class TwoStageEstimator(BaseEstimator):
    '''
    '''
    #region: __init__
    def __init__(
            self, 
            stage1_estimator,  # classifier
            stage2_estimator,  # regressor
            target_transform=np.log10,
            target_inverse_transform=lambda x: 10**x
            ):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.target_transform = target_transform
        self.target_inverse_transform = target_inverse_transform
    #endregion

    #region: fit
    def fit(self, X, y, groups_stage2=None):
        '''
        '''
        X = np.asarray(X)
        y = np.asarray(y)

        self._fit_stage1(X, y)
        self._fit_stage2(X, y, groups_stage2=groups_stage2)

        return self
    #endregion

    #region: _fit_stage1
    def _fit_stage1(self, X, y):
        '''
        '''
        detection = (y > 0).astype(int)
        self.stage1_estimator.fit(X, detection)
    #endregion
    
    #region: _fit_stage2
    def _fit_stage2(self, X, y, groups_stage2=None):
        '''
        '''
        where_detected = y > 0 
        if where_detected.sum() == 0:
            raise ValueError('No detected samples to fit stage-2.')
        X_det = X[where_detected]
        y_det = y[where_detected]

        y_det_trans = self.target_transform(y_det)

        fit_params = self._prepare_fit_params(groups_stage2, where_detected)
        
        self.stage2_estimator.fit(X_det, y_det_trans, **fit_params)
    #endregion

    #region: _prepare_fit_params
    def _prepare_fit_params(self, groups_stage2, where_detected):
        '''
        Prepare the fit parameters for the stage-2 estimator if it requires a
        grouping variable (e.g., if it is a MixedLMRegressor).

        Parameters
        ----------
        groups_stage2 : array-like
            Grouping variable for the random intercept.
        where_detected : array-like of bool
            Boolean mask indicating the detected samples.

        Returns
        -------
        dict
            Fit parameters to be passed to the estimator's fit method.
        '''
        final_est = self.stage2_estimator
        if hasattr(self.stage2_estimator, 'steps'):
            # final_estimator is embedded in a pipeline
            final_est = self.stage2_estimator.steps[-1][1]

        fit_params = {}
        if (groups_stage2 is not None and 
                isinstance(final_est, MixedLMRegressor)):
            groups_det = np.asarray(groups_stage2)[where_detected]
            if hasattr(self.stage2_estimator, 'steps'):
                # final_estimator is embedded in a pipeline
                fit_params = {'regressor__groups': groups_det}
            else:
                fit_params = {'groups': groups_det}
        return fit_params
    #endregion
         
    #region: predict
    def predict(self, X):
        '''
        '''
        X = np.asarray(X)

        detect_pred = self.stage1_estimator.predict(X)
        y_pred = np.zeros(X.shape[0])  # initialize

        where_detected = detect_pred == 1
        if where_detected.sum() > 0:
            X_det = X[where_detected]
            y_pred_trans = self.stage2_estimator.predict(X_det)
            y_pred[where_detected] = self.target_inverse_transform(y_pred_trans)

        return y_pred
    #endregion
#endregion

###############################################################################
# 4. Cross-Validation Helpers
###############################################################################

# TODO: Clarify that groups_stage2 is for mixed LM
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
    y = np.asarray(y)

    fold_results = []
    for train_idx, test_idx in cv.split(X, y, groups=groups_cv):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if groups_stage2 is not None:
            groups_stage2_train = np.asarray(groups_stage2)[train_idx]
        else:
            groups_stage2_train = None

        estimator.fit(X_train, y_train, groups_stage2=groups_stage2_train)

        metrics = _evaluate_fold(
            estimator, 
            X_test, 
            y_test,
            clf_funcs, 
            reg_funcs
            )
        fold_results.append(metrics)
        
    return fold_results
#endregion

#region: _evaluate_fold
def _evaluate_fold(
        estimator, 
        X_test, 
        y_test, 
        clf_funcs, 
        reg_funcs
        ):
    '''
    '''
    y_pred = estimator.predict(X_test)

    ## Evaluate stage-1 classification
    # Attempt to obtain probability estimates for AUC
    try:
        y_proba = estimator.stage1_estimator.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None
    # Convert continuous 'y' back to binary detect/nondetect
    clf_metrics = _classification_metrics(
        (y_test > 0).astype(int),
        (y_pred > 0).astype(int),
        clf_funcs,
        y_proba=y_proba
    )

    ## Evaluate stage-2 regression
    reg_metrics = _regression_metrics(
        y_test, 
        y_pred, 
        reg_funcs,
        estimator.target_transform
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
            # FIXME: Hardcoded 'auc'
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
    mask = (y_true > 0) & (y_pred > 0)  # true positives
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

# TODO: Specify parameters in config
# TODO: Expand the docstring with more explanation
#region: holdout_chemicals
def holdout_chemicals(
        y, 
        chem_groups, 
        holdout_fraction=0.1,
        random_state=42
        ):
    '''Create hold-out sets by chemical.'''
    unique_chems = np.unique(chem_groups)
    n_holdout = int(len(unique_chems) * holdout_fraction)
    rng = np.random.RandomState(random_state)
    holdout_set = rng.choice(unique_chems, size=n_holdout, replace=False)
    mask = np.isin(chem_groups, holdout_set)
    return y[~mask], y[mask], ~mask, mask
#endregion
 
###############################################################################
# 6. Main Execution
###############################################################################

#region: __main__
if __name__ == '__main__':

    # TODO: Move to config. Compare with POD repo structure
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