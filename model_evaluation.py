'''
'''

import pandas as pd 
import numpy as np 

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

    performances = []
    for train_idx, test_idx in cv.split(X, y, groups=groups_cv):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if groups_stage2 is not None:
            groups_stage2_train = np.asarray(groups_stage2)[train_idx]
        else:
            groups_stage2_train = None

        estimator.fit(X_train, y_train, groups_stage2=groups_stage2_train)

        fold_performance = _evaluate_fold(
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
    
    fold_performance = {}
    fold_performance.update(clf_metrics)
    fold_performance.update(reg_metrics)

    return fold_performance
#endregion

#region: _classification_metrics
def _classification_metrics(y_true, y_pred, metric_funcs, y_proba=None):
    '''
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

# TODO: Expand the docstring with more explanation
#region: holdout_chemicals
def holdout_chemicals(
        y, 
        chem_groups, 
        holdout_fraction=0.1,
        random_state=42
        ):
    '''
    Create hold-out sets by chemical.
    '''
    unique_chems = np.unique(chem_groups)
    n_holdout = int(len(unique_chems) * holdout_fraction)
    rng = np.random.RandomState(random_state)
    holdout_set = rng.choice(unique_chems, size=n_holdout, replace=False)
    mask = np.isin(chem_groups, holdout_set)
    return y[~mask], y[mask], ~mask, mask
#endregion