'''
'''

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

def regression_analysis(X, y):
    """
    Perform regression analysis using a RandomForestRegressor with target transformation.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.

    Returns:
    dict: A dictionary containing the R2, RMSE, and MedAE of the model.
    """
    X, y = ensure_data_integrity(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=1
    )

    feature_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    regressor = TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=42),
        transformer=PowerTransformer(method='yeo-johnson')
    )

    model = Pipeline([
        ('features', feature_pipeline),
        ('regressor', regressor)
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        'X_train' : X_train,
        'X_test' : X_test, 
        'y_train' : y_train,
        'y_test' : y_test,
        'y_pred' : y_pred
    }

    return results

def evaluate_regression(y_test, y_pred):
    '''
    '''
    performances = {
        'R2' : r2_score(y_test, y_pred),
        'RMSE' : mean_squared_error(y_test, y_pred, squared=False),
        'MedAE' : median_absolute_error(y_test, y_pred)
    }

    return performances

def classification_analysis(X, y):
    '''
    '''
    X, y = ensure_data_integrity(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=1
    )

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        'X_train' : X_train,
        'X_test' : X_test, 
        'y_train' : y_train,
        'y_test' : y_test,
        'y_pred' : y_pred
    }

    return results

def ensure_data_integrity(X, y):
    '''
    '''
    y = y.dropna()
    X, y = with_common_index(X, y)
    return X, y

def with_common_index(X, y):
    '''
    '''
    common_samples = list(X.index.intersection(y.index))
    return X.loc[common_samples], y.loc[common_samples]