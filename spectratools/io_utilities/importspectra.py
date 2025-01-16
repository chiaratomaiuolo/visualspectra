"""Utilities for importing spectra from various file formats.
   The avaliable formats (by now) are:
   - .txt [x]
   - .csv [x]
   - .root [x]
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
    data = np.loadtxt(file_path)
    if data.shape[1] == 1: # This is the case of GammaStream MC2 acquired histo
        bins, content = np.arange(data.shape[0]), data
    elif data.shape[1] == 2: # This is a standard bin, content format
        bins, content = data[:,0], data[:,1]
    elif data.shape[1] == 3: # This is a bin, content, error format
        bins, content = data[:,0], data[:,1]
    else: # This is an unknown format
        print("Unknown format.")
        sys.exit(1)
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
    data = np.loadtxt(file_path, delimiter=',')

    if data.shape[1] == 1: # This is the case of GammaStream MC2 acquired histo
        bins, content = np.arange(data.shape[0]), data
    elif data.shape[1] == 2: # This is a standard bin, content format
        bins, content = data[:,0], data[:,1]
    elif data.shape[1] == 3: # This is a bin, content, error format
        bins, content = data[:,0], data[:,1]
    else: # This is an unknown format
        print("Unknown format.")
        sys.exit(1)
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

def import_spectrum(file_path: str | os.PathLike) -> Tuple[np.array, np.array]:
    if file_path.endswith('.txt'):
        return import_from_txt(file_path)
    elif file_path.endswith('.csv'):
        return import_from_csv(file_path)
    elif file_path.endswith('.root'):
        return import_from_root(file_path)
    else:
        print("File format not supported.")
        sys.exit(1)

