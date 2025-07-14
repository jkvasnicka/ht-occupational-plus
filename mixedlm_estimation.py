'''
Defines MixedLMRegressor, a scikit-learn compatible estimator for fitting
mixed-effects linear models with random intercepts via statsmodels' formula 
API.
'''

import numpy as np 
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

#region: MixedLMRegressor
class MixedLMRegressor(BaseEstimator, RegressorMixin):
    '''
    MixedLMRegressor wrapper for statsmodels MixedLM.
    '''
    def __init__(self, method='bfgs', reml=False, maxiter=200):
        '''
        See statsmodels documentation for hyperparameter definitions.
        '''
        self.method = method
        self.reml = reml
        self.maxiter = maxiter

    #region: fit
    def fit(self, X, y, groups=None):
        '''
        Fit the mixed-effects linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target vector.
        groups : array-like of shape (n_samples,)
            Group labels for random intercepts.

        Returns
        -------
        self : MixedLMRegressor
            Fitted estimator instance.
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
        Predict target (y) using the fitted mixed-effects model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        '''
        X_df = self._ensure_dataframe(np.asarray(X))
        return self.result_.predict(X_df)
    #endregion

    #region: get_icc
    def get_icc(self):
        '''
        Compute intraclass correlation coefficient (ICC) from fitted model.

        Returns
        -------
        float
        '''
        check_is_fitted(self, 'result_')

        if self.result_.cov_re.shape[0] > 0:
            var_group = self.result_.cov_re.iloc[0, 0]
        else:
            var_group = 0.0
        var_resid = self.result_.scale
        
        return MixedLMRegressor.intraclass_correlation(var_group, var_resid)
    #endregion

    #region: intraclass_correlation
    @staticmethod
    def intraclass_correlation(var_group, var_resid):
        '''
        Computes Intraclass Correlation Coefficient (ICC).
        
        Parameters
        ----------
        var_group :
            Random intercept variance (mdf.cov_re.iloc[0,0]).
        var_resid : 
            Residual variance (mdf.scale).

        Returns
        -------
        float
        '''
        return var_group / (var_group + var_resid)
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
        Prepare DataFrame and formula for mixed-effects model fitting.

        Returns
        -------
        Xy_df : DataFrame
            Combined DataFrame with predictor columns and target 'y'.
        formula : str
            Patsy formula string for fixed effects (e.g., 'y ~ x0 + x1 + ...')
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