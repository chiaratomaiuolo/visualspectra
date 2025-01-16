"""Utilities for importing spectra from various file formats.
   The avaliable formats (by now) are:
   - .txt
   - .csv
   - .root 
"""

# System libraries first...
import os
import sys
from typing import Tuple
# ... then installed libraries...
import numpy as np
# ... and eventually local modules
import uproot

def import_from_txt(file_path: str | os.PathLike) -> Tuple[np.array, np.array]:
    """Import a spectrum from a .txt file.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to the .txt file.

    Returns
    -------
    bins, content : Tuple[np.array, np.array]
        Numpy arrays containing the bin limits and their content.
    """
    try:
        bins, content = np.loadtxt(file_path, unpack=True)
        return bins, content
    except ValueError:
        bins, content, error = np.loadtxt(file_path, unpack=True)
        return bins, content

def import_from_csv(file_path: str | os.PathLike) -> Tuple[np.array, np.array]:
    """Import a spectrum from a .csv file.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to the .txt file.

    Returns
    -------
    bins, content : Tuple[np.array, np.array]
        Numpy arrays containing the bin limits and their content.
    """
    try:
        bins, content = np.loadtxt(file_path, delimiter=',', unpack=True)
        return bins, content
    except ValueError:
        bins, content, error = np.loadtxt(file_path, unpack=True)
        return bins, content
    
def import_from_root(file_path: str | os.PathLike, treename: str='Data_R') -> Tuple[np.array, np.array]:
    """Import a spectrum from a .root file. The default name of the tree is 'Data_R',
    the one given in a CoMPASS acquisition to the raw sampled data.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to the .root file
    treename : str
        Name of the tree containing the data. Default is 'Data_R'.
    
    Returns
    -------
    hist, bin_edges : Tuple[np.array, np.array]
        Numpy arrays containing the bin limits and their content.
    """
    # Opening file
    rootfile = uproot.open(file_path)
    # Getting the tree containing data
    tree = rootfile[treename]
    # Getting the spectrum
    spectrum = tree['Energy'].array()
    # Converting to np.arrays...
    spectrum = np.array(spectrum) 
    # ... creating the histogram.
    # Default number of bins is 4096 (CoMPASS acquisition)
    hist, bin_edges = np.histogram(spectrum, bins=4096)
    return hist, bin_edges

