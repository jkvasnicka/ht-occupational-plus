'''
'''

import numpy as np 
from sklearn.base import BaseEstimator

from mixedlm_estimation import MixedLMRegressor

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