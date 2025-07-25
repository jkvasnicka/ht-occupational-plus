'''
This module provides functions to perform k-fold cross-validation and
holdout evaluation for a TwoStageEstimator, compute classification and
regression metrics, and handle data partitioning by chemical groups.
'''

import pandas as pd 
import numpy as np 
from sklearn.model_selection import GroupKFold

from metrics_management import metrics_from_config
import results_management

#region: evaluate_twostage
def evaluate_twostage(
        estimator,
        X_full, 
        y_full,
        config,
        evaluation_type
        ):
    '''
    Orchestrates cross-validation or holdout evaluation and persist results.

    Parameters
    ----------
    estimator : TwoStageEstimator
        The two-stage estimator to evaluate.
    X_full : array-like, shape (n_samples, n_features)
        Full feature matrix.
    y_full : pandas.Series
        Target values with MultiIndex including 'DTXSID' and 'naics_id'.
    config : UnifiedConfiguration
        Configuration object with data splitting and evaluation parameters.
    evaluation_type : {'cv', 'holdout'}
        'cv' to run k-fold cross-validation; 'holdout' to evaluate on a 
        validation set.

    Returns
    -------
    None

    Notes
    -----
    Writes performance metrics CSV and (for holdout) the fitted estimator.
    '''
    chem_groups = y_full.index.get_level_values('DTXSID')
    naics_groups = y_full.index.get_level_values('naics_id')
    
    X_full = np.asarray(X_full)
    y_full = np.asarray(y_full)

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

    clf_funcs = metrics_from_config(config.metrics['classification'])
    reg_funcs = metrics_from_config(config.metrics['regression'])
    
    if evaluation_type == 'cv':
        print('  Cross-validating two-stage estimator.')

        cv = GroupKFold(n_splits=config.data['n_splits_cv'])

        cv_performances = cross_validate_twostage(
            estimator, 
            X_dev, 
            y_dev, 
            cv, 
            chem_groups_dev, 
            clf_funcs, 
            reg_funcs, 
            groups_stage2=groups_stage2
            )
        
        results_management.write_performance(
            cv_performances, 
            config.path['results_dir'], 
            config.file,
            'cv_performances.csv'
            )

    elif evaluation_type == 'holdout':
        print('  Evaluating hold-out performance.')

        holdout_performance, holdout_pred, estimator = (
            evaluate_holdout_performance(
                estimator, 
                X_dev, 
                y_dev,
                groups_stage2,
                X_val, 
                y_val, 
                clf_funcs, 
                reg_funcs,
                X_full,
                y_full,
                naics_groups
            )
        )

        results_management.write_performance(
            holdout_performance, 
            config.path['results_dir'], 
            config.file,
            'holdout_performance.csv'
        )

        results_management.write_estimator(
            estimator, 
            config.path['results_dir'], 
            config.file
        )
#endregion

#region: evaluate_holdout_performance
def evaluate_holdout_performance(
        estimator, 
        X_dev, 
        y_dev,
        groups_stage2,
        X_val, 
        y_val, 
        clf_funcs, 
        reg_funcs,
        X_full,
        y_full,
        naics_groups
        ):
    '''
    Fit on dev set, evaluate on holdout, refit on full data, and return 
    results.

    Parameters
    ----------
    estimator : TwoStageEstimator
        Instantiated estimator.
    X_dev : array-like
        Features for development (training).
    y_dev : array-like
        Targets for development.
    groups_stage2 : array-like or None
        Group labels for mixed-effects stage or None.
    X_val : array-like
        Features for the holdout set.
    y_val : array-like
        Targets for the holdout set.
    clf_funcs : dict
        Classification metric functions.
    reg_funcs : dict
        Regression metric functions.
    X_full : array-like
        Full feature matrix for final refit.
    y_full : array-like
        Full target vector for final refit.
    naics_groups : array-like
        Group labels for full data refit if MixedLMRegressor is used.

    Returns
    -------
    holdout_performance : pandas.Series
        Metrics indexed by (stage, metric) for the holdout set.
    holdout_pred : pandas.Series
        Predicted values on the holdout set.
    estimator : TwoStageEstimator
        Estimator refitted on the full dataset.
    '''
    estimator.fit(X_dev, y_dev, groups_stage2=groups_stage2)

    holdout_performance, holdout_pred = _evaluate_performance(
        estimator, 
        X_val, 
        y_val, 
        clf_funcs, 
        reg_funcs
        )
    holdout_performance = pd.Series(holdout_performance)
    holdout_performance.index.names = ['stage', 'metric']
    # FIXME: Need to preserve the original index
    holdout_pred = pd.Series(holdout_pred)
            
    # Refit the estimator on the full dataset
    estimator.fit(X_full, y_full, groups_stage2=naics_groups)

    return holdout_performance, holdout_pred, estimator
#endregion

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
    Perform grouped k-fold cross-validation for a TwoStageEstimator.

    Parameters
    ----------
    estimator : TwoStageEstimator
        The two-stage estimator to evaluate.
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target vector.
    cv : cross-validation splitter
        e.g., GroupKFold instance.
    groups_cv : array-like, shape (n_samples,)
        Group labels for splitting (chemical IDs).
    clf_funcs : dict
        Classification metric functions.
    reg_funcs : dict
        Regression metric functions.
    groups_stage2 : array-like or None
        Group labels for mixed-effects stage or None if not used.

    Returns
    -------
    performances : pandas.DataFrame
        DataFrame of fold metrics with MultiIndex columns ('stage', 'metric').
    '''
    X = np.asarray(X)
    y = np.asarray(y)

    performances = []
    for train_idx, test_idx in cv.split(X, y, groups=groups_cv):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if groups_stage2 is not None:
            groups_stage2_train = np.asarray(groups_stage2)[train_idx]
        else:
            groups_stage2_train = None

        estimator.fit(X_train, y_train, groups_stage2=groups_stage2_train)

        fold_performance, *_ = _evaluate_performance(
            estimator, 
            X_test, 
            y_test,
            clf_funcs, 
            reg_funcs
            )
        performances.append(fold_performance)

    performances = pd.DataFrame(performances)
    performances.columns = pd.MultiIndex.from_tuples(performances.columns)
    performances.columns.names = ['stage', 'metric']
        
    return performances
#endregion

#region: _evaluate_performance
def _evaluate_performance(
        estimator, 
        X_test, 
        y_test, 
        clf_funcs, 
        reg_funcs
        ):
    '''
    Compute classification and regression metrics for a single test set.
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
    
    performance = {}
    performance.update(clf_metrics)
    performance.update(reg_metrics)

    return performance, y_pred
#endregion

#region: _classification_metrics
def _classification_metrics(y_true, y_pred, metric_funcs, y_proba=None):
    '''
    Compute binary classification metrics, using probabilities for AUC.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels.
    metric_funcs : dict
        Mapping metric names to functions.
    y_proba : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    metrics : dict
        Mapping (stage1, metric) to computed value.
    '''
    metrics = {}
    for k, func in metric_funcs.items():
        k = ('stage1', k)
        try:
            if k == 'roc_auc_score' and y_proba is not None:
                metrics[k] = func(y_true, y_proba)
            else:
                metrics[k] = func(y_true, y_pred)
        except Exception:
            metrics[k] = np.nan
    return metrics
#endregion

#region: _regression_metrics
def _regression_metrics(y_true, y_pred, metric_funcs, target_transform):
    '''
    Compute regression metrics on positive predictions only.

    Parameters
    ----------
    y_true : array-like
        True continuous values.
    y_pred : array-like
        Predicted continuous values.
    metric_funcs : dict
        Mapping metric names to functions.
    target_transform : callable
        Function used to transform y for metric computation (e.g., log10).

    Returns
    -------
    metrics : dict
        Mapping (stage2, metric) to computed value.
    '''
    mask = (y_true > 0) & (y_pred > 0)  # true positives
    metrics = {}
    if np.sum(mask) > 0:
        y_true_trans = target_transform(y_true[mask])
        y_pred_trans = target_transform(y_pred[mask])
        for k, func in metric_funcs.items():
            k = ('stage2', k)
            try:
                metrics[k] = func(y_true_trans, y_pred_trans)
            except Exception:
                metrics[k] = np.nan
    else:
        for k in metric_funcs.keys():
            k = ('stage2', k)
            metrics[k] = np.nan
    return metrics
#endregion

#region: holdout_chemicals
def holdout_chemicals(
        y, 
        chem_groups, 
        holdout_fraction=0.1,
        random_state=42
        ):
    '''
    Split data into development and holdout sets by chemical group.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Target vector.
    chem_groups : array-like, shape (n_samples,)
        Chemical identifiers for grouping.
    holdout_fraction : float, optional
        Fraction of unique chemicals to hold out.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    y_dev : ndarray
        Targets for development set.
    y_val : ndarray
        Targets for holdout set.
    dev_mask : ndarray of bool
        Mask selecting development samples.
    val_mask : ndarray of bool
        Mask selecting holdout samples.
    '''
    y = np.asarray(y)
    unique_chems = np.unique(chem_groups)
    n_holdout = int(len(unique_chems) * holdout_fraction)
    rng = np.random.RandomState(random_state)
    holdout_set = rng.choice(unique_chems, size=n_holdout, replace=False)
    mask = np.isin(chem_groups, holdout_set)
    return y[~mask], y[mask], ~mask, mask
#endregion