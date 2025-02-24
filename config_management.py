'''
This module contains the `UnifiedConfiguration` class, which centralizes the 
management and access of configuration settings related to different categories,
such as paths, models, and plotting. By unifying the configuration into a 
single object, it allows for a more streamlined access to settings throughout 
the code.

Example
-------
config_file = 'config.json'
config = UnifiedConfiguration(config_file)
model_settings = config.model
'''

import json
import argparse

#region: UnifiedConfiguration
class UnifiedConfiguration:
    '''
    This class provides a unified interface to access configuration settings 
    related to different categories such as paths, models, and plotting. The 
    configuration files for each category are loaded and made accessible as 
    attributes.
    '''
    #region: __init__
    def __init__(self, config_file, encoding=None):
        '''
        Initialize the UnifiedConfiguration object.

        Parameters
        ----------
        config_file : str
            Path to the JSON file mapping categories to configuration file 
            paths. By default, will look for 'config.json' in the working 
            directory.
        encoding : str, optional
            Default is 'utf-8'.
        '''
        self.file = config_file
        
        if encoding is None:
            encoding = 'utf-8'  # default

        with open(config_file, 'r', encoding=encoding) as mapping_file:
            config_files_dict = json.load(mapping_file)

        # Load each file into its category
        for category, file_path in config_files_dict.items():
            with open(file_path, 'r', encoding=encoding) as config_file:
                setattr(self, category, json.load(config_file))
    #endregion
#endregion

#region: base_cli_parser
def base_cli_parser():
    '''
    Create a base parser with arguments needed for configuration loading.

    Returns
    -------
    argparse.ArgumentParser
        A parser with base configuration arguments.
    '''
    # Only the child parser supplies help to avoid conflict
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        help='Path to the main configuration file',
        default=None
    )
    parser.add_argument(
        '-e',
        '--encoding',
        type=str,
        help='Encoding of the configuration files',
        default=None
    )
    return parser
#endregion