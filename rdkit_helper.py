'''
'''

import pandas as pd
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def get_2d_descriptors(
        smiles_for_chem, 
        index_name,
        write_path=None
        ):
    '''
    Get all two-dimensional molecular descriptors from RDKit.

    Parameters
    ----------
    smiles_for_chem : dict
        Mapping of DTXSID to SMILES string.
    index_name : str
        Used to name the index, e.g., 'DTXSID'.
    write_path : str (optional)
        Path to write the return as a parquet file.
    
    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    RDKit also provides some more recent "2D-autocorrelation" descriptors, 
    but the functions may have not been fully debugged.
    '''
    mol_for_chem = {
        chem: MolFromSmiles(smiles) 
        for chem, smiles in smiles_for_chem.items()}

    # Filter out None values which correspond to parsing errors.
    mol_for_chem = {
        chem: mol for chem, mol in mol_for_chem.items() 
        if mol is not None}

    desc_list = [tup[0] for tup in Descriptors._descList]
    calc = (
        MoleculeDescriptors
        .MolecularDescriptorCalculator(desc_list))

    descriptors = pd.DataFrame.from_dict(
        {chem: calc.CalcDescriptors(mol) 
        for chem, mol in mol_for_chem.items()}, 
        orient='index',
        columns=calc.GetDescriptorNames())
    descriptors.index.name = index_name
        
    if write_path is not None:
        descriptors.to_parquet(write_path)

    return descriptors