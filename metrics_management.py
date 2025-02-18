'''
'''

import importlib

#region: metrics_from_config
def metrics_from_config(metrics_settings):
    '''
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
    '''
    return lambda y_true, y_pred: f(y_true, y_pred, **kwargs)
#endregion