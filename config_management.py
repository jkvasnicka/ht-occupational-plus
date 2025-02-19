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

#region: UnifiedConfiguration.__init__
class UnifiedConfiguration:
    '''
    This class provides a unified interface to access configuration settings 
    related to different categories such as paths, models, and plotting. The 
    configuration files for each category are loaded and made accessible as 
    attributes.
    '''
    def __init__(self, config_file=None, encoding=None):
        '''
        Initialize the UnifiedConfiguration object.

        Parameters
        ----------
        config_file : str, optional
            Path to the JSON file mapping categories to configuration file 
            paths. By default, will look for 'config.json' in the working 
            directory.
        encoding : str, optional
            Default is 'utf-8'.
        '''
        # Set defaults
        if config_file is None:
            config_file = 'config.json'  
        if encoding is None:
            encoding = 'utf-8'

        with open(config_file, 'r', encoding=encoding) as mapping_file:
            config_files_dict = json.load(mapping_file)

        # Load each file into its category
        for category, file_path in config_files_dict.items():
            with open(file_path, 'r', encoding=encoding) as config_file:
                setattr(self, category, json.load(config_file))
#endregion

#region: parse_args
def parse_args():
    '''
    Parse command-line arguments for configuration loading

    The function defines and parses two command-line arguments:
    - `config_file`: An optional positional argument specifying the path to 
        the main configuration file.
    - `encoding`: An optional argument specifying the encoding of the 
        configuration file.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    '''
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()
#endregion

#region: config_from_cli_args
def config_from_cli_args():
    '''
    Load the configuration using command-line interface arguments.

    Returns
    -------
    UnifiedConfiguration
        An instance of UnifiedConfiguration initialized with the parsed 
        command-line arguments.
    '''
    args = parse_args()
    return UnifiedConfiguration(
        config_file=args.config_file, 
        encoding=args.encoding
        )
#endregion