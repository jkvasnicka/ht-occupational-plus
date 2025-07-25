'''
This module provides functions to build scikit-learn Pipelines for both
stage-1 (classification) and stage-2 (regression) steps, and to assemble
a TwoStageEstimator instance from model configuration settings (dict).

The configuration settings must contain two keys, 'stage1' and 'stage2', each
mapping to a list of step settings with:
  - 'name'   : str, name of the pipeline step
  - 'module' : str, import path of the estimator module
  - 'class'  : str, estimator class name within the module
  - 'kwargs' : dict, optional parameters for the estimator
'''

import importlib
from sklearn.pipeline import Pipeline

import twostage_estimation

#region: twostage_estimator_from_config
def twostage_estimator_from_config(
        model_settings,
        target_transform=twostage_estimation.transform_log10,
        target_inverse_transform=twostage_estimation.inverse_log10
        ):
    '''
    Build a TwoStageEstimator from model configuration settings.

    Parameters
    ----------
    model_settings : dict
        Contains two keys: 'stage1' and 'stage2', each with a list defining 
        the pipeline for that stage.
    target_transform : callable, optional
        Function to transform the target variable (default: np.log10).
    target_inverse_transform : callable, optional
        Function to invert the target transformation (default: lambda x: 10**x).

    Returns
    -------
    TwoStageEstimator
        A TwoStageEstimator instance with the specified stage1 and stage2 
        pipelines.
    '''
    stage1_config = model_settings['stage1']
    stage2_config = model_settings['stage2']
    stage1_pipeline = pipeline_from_config(stage1_config)
    stage2_pipeline = pipeline_from_config(stage2_config)
    
    return twostage_estimation.TwoStageEstimator(
        stage1_estimator=stage1_pipeline,
        stage2_estimator=stage2_pipeline,
        target_transform=target_transform,
        target_inverse_transform=target_inverse_transform
        )
#endregion

#region: pipeline_from_config
def pipeline_from_config(steps_config):
    '''
    Build a scikit-learn Pipeline from a list of step configurations.

    Parameters
    ----------
    steps_config : list of dict
        Each dict should have keys 'name', 'module', 'class', and optionally
        'kwargs'.

    Returns
    -------
    Pipeline
        The assembled Pipeline.
    '''
    steps = []
    for step in steps_config:
        name = step['name']
        estimator = estimator_from_config(step)
        steps.append((name, estimator))
    return Pipeline(steps)
#endregion

#region: estimator_from_config
def estimator_from_config(config):
    '''
    Instantiate an estimator from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Must contain 'module', 'class', and optionally 'kwargs'.

    Returns
    -------
    estimator : object
        Instantiated estimator.
    '''
    module = importlib.import_module(config['module'])
    est_class = getattr(module, config['class'])
    params = config.get('kwargs', {})
    return est_class(**params)
#endregion
