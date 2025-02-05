'''
'''

import pandas as pd 

#region: data_from_raw
def data_from_raw(comptox_file):
    '''
    Load and clean data from the CompTox Chemistry Dashboard.

    Parameters
    ----------
    comptox_file : str
        Path to the CSV file containing raw CompTox data.

    Returns
    -------
    pd.DataFrame
    '''
    comptox_data = pd.read_csv(comptox_file)

    comptox_data = comptox_data.drop_duplicates()

    # comptox_data = resolve_chemical_ambiguities(
    #         comptox_data, 
    #         found_by_col, 
    #         duplicate_warning, 
    #         rank_map, 
    #         substance_input_col
    # )
        
    return comptox_data
#endregion

# NOTE: Deprecated?
#region: resolve_chemical_ambiguities
def resolve_chemical_ambiguities(
        comptox_data, 
        found_by_col, 
        duplicate_warning, 
        rank_map, 
        substance_input_col
        ):
    '''
    Resolve ambiguities in chemical input mappings to DTXSIDs.

    Ambiguous matches are removed, and the highest-ranked suggestion is 
    retained based on the inputted rank map.

    Parameters
    ----------
    found_by_col : str
        Name of the column indicating the source of the match.
    duplicate_warning : str
        Text suffix to remove from values in `found_by_col`.
    rank_map : dict
        Mapping of values in `found_by_col` to rank scores. Higher values 
        indicate greater priority.
    substance_input_col : str
        Name of the column containing input chemical names.

    Notes
    -----
    This function was designed to handle situations where the CompTox input
    column is the raw substance name, i.e., we used CompTox to get the likely
    DTXSIDs. However, since then, a direct mapping from raw name to DTXSID has
    become available, eliminating the need for this function. This function 
    may therefore be deprecated but is left here for future use.
    '''
    comptox_data = comptox_data.copy()

    # Remove the redundant warning message so that the mapping doesn't have 
    # to include it
    comptox_data[found_by_col] = (
        comptox_data[found_by_col]
        .str.replace(duplicate_warning, '')
        .str.strip()
    )
    # Define a temporary column to rank suggested chemical names
    rank_col = 'RANK'
    comptox_data[rank_col] = (
        comptox_data[found_by_col]
        .map(rank_map)
    )

    # Remove any chemicals with several matches but equally ranked
    where_ambiguous = comptox_data.duplicated(
        subset=[substance_input_col, rank_col], 
        keep=False
        )
    comptox_data = comptox_data.drop(
        comptox_data.loc[where_ambiguous].index
        )

    comptox_data = (
        comptox_data
        .sort_values(
            by=[substance_input_col, rank_col], 
            ascending=[True, False]
            )
        .drop_duplicates(subset=substance_input_col, keep='first')
    )

    return comptox_data.drop(rank_col, axis=1)
#endregion