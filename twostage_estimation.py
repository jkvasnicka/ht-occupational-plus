'''
Defines the TwoStageEstimator class, which first classifies non-zero 
detections then regresses on log10-transformed positive targets. 

Supports scikit-learn estimators and a MixedLMRegressor via grouping.
'''

import numpy as np 
from sklearn.base import BaseEstimator

from mixedlm_estimation import MixedLMRegressor

# Defining named functions allows joblib to serialize the TwoStageEstimator
def transform_log10(x):
    return np.log10(x)

def inverse_log10(x):
    return 10**x

#region: TwoStageEstimator
class TwoStageEstimator(BaseEstimator):
    '''
    Two-stage estimator combining classification and regression.

    First fits a classifier to predict detection (y>0), then fits a regressor
    on the transformed positive targets. Can pass grouping labels to the
    regressor for mixed-effects models.
    '''
    #region: __init__
    def __init__(
            self, 
            stage1_estimator,  # classifier
            stage2_estimator,  # regressor
            target_transform=transform_log10,
            target_inverse_transform=inverse_log10
            ):
        '''
        Initialize the two-stage estimator.

        Parameters
        ----------
        stage1_estimator : estimator
            Classifier for detection (binary).
        stage2_estimator : estimator
            Regressor for positive targets, possibly wrapped in a Pipeline.
        target_transform : callable, default=transform_log10
            Function to apply to positive targets before regression.
        target_inverse_transform : callable, default=inverse_log10
            Function to invert regression output back to original scale.
        '''        
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.target_transform = target_transform
        self.target_inverse_transform = target_inverse_transform
    #endregion

    #region: fit
    def fit(self, X, y, groups_stage2=None):
        '''
        Fit the detection and regression stages.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target vector with zeros for nondetects.
        groups_stage2 : array-like, optional
            Group labels for mixed-effects regressor.

        Returns
        -------
        self : TwoStageEstimator
            Fitted estimator instance.
        '''
        X = np.asarray(X)
        y = np.asarray(y)

        self._fit_stage1(X, y)
        self._fit_stage2(X, y, groups_stage2=groups_stage2)

        return self
    #endregion

    #region: _fit_stage1
    def _fit_stage1(self, X, y):
        '''Fit the binary detection classifier.'''
        detection = (y > 0).astype(int)
        self.stage1_estimator.fit(X, detection)
    #endregion
    
    #region: _fit_stage2
    def _fit_stage2(self, X, y, groups_stage2=None):
        '''Fit the regression model on detected samples.'''
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
        Predict target values with two-stage logic.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted values: zeros for nondetects, inverse-transformed
            regression predictions for detected samples.
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