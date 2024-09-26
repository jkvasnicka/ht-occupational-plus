'''
'''

import numpy as np

# TODO: Drop samples if unit not M or P for consistency with USIS?
# NOTE: Measure unit IDs are NaN if exposure level is null
#region: prepare_concentration_target
def prepare_concentration_target(sample_results, measure_units, mol_weights):
    '''
    Prepare the target variable of chemical concentration in air with a 
    consistent unit of mg/m3.
    '''
    return np.where(
        measure_units=='M',  # already mg/m3
        sample_results,
        np.where(
            measure_units=='P',  # ppm
            ppm_to_mg_m3(sample_results, mol_weights),
            np.nan  # everything else
        )
    )
#endregion

#region: ppm_to_mg_m3
def ppm_to_mg_m3(ppm, mw):
    '''
    Convert chemical concentration unit from parts per million to milligram 
    per cubic meter.

    Parameters
    ----------
    ppm : float or array-like
        Value in PPM.
    mw : float or array-like
        Molecular weight [g/mol].

    Reference
    ---------
    https://www.ccohs.ca/oshanswers/chemicals/convert.html
    '''
    return ppm * mw/24.45
#endregion