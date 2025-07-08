'''
Provides utilities to dynamically import metric functions by name
and instantiate them for use in model evaluation.
'''

import importlib

#region: metrics_from_config
def metrics_from_config(metrics_settings):
    '''
    Instantiate metric callables based on configuration.

    Parameters
    ----------
    metrics_settings : dict
        Mapping from metric name to dict with keys:
        - 'module': module path (str)
        - 'class': function name (str), defaults to name key
        - 'kwargs': optional dict of keyword arguments for the function

    Returns
    -------
    dict
        Mapping from metric name to a callable f(y_true, y_pred).
    '''
    function_for_metric = {}
    for name, config in metrics_settings.items():
        module = importlib.import_module(config['module'])
        class_name = config.get('class', name)
        kwargs = config.get('kwargs', {})
        f = getattr(module, class_name)
        function_for_metric[name] = build_scoring_function(f, kwargs)
    return function_for_metric
#endregion

#region: build_scoring_function
def build_scoring_function(f, kwargs):
    '''
    Wrap a metric function with fixed keyword arguments.

    Parameters
    ----------
    f : callable
        Metric function expecting signature f(y_true, y_pred, **kwargs).
    kwargs : dict
        Keyword arguments to bind to `f`.

    Returns
    -------
    callable
        Function scoring(y_true, y_pred) that calls f with bound kwargs.
    '''
    return lambda y_true, y_pred: f(y_true, y_pred, **kwargs)
#endregion