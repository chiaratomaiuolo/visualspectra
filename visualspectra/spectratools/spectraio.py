"""Utilities for importing spectra from various file formats.
   The avaliable formats (by now) are:
   - .txt [x]
   - .csv [x]
   - .root [x]
"""

# System libraries first...
import argparse
import os
import sys
from typing import Tuple
# ... then installed libraries...
import numpy as np
import uproot
# ... and eventually local modules


def create_parser() -> argparse.ArgumentParser:
    """ Create the parser for the command line arguments.

    Return
    ------
    parser : argparse.ArgumentParser
        The parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Spectrum Plotter Application')
    parser.add_argument('--filepaths', type=str, nargs='+',\
                        help='Paths to the files to be plotted')
    parser.add_argument('--nbins', type=int, default=1024,\
                        help='Number of bins of the spectrum')
    parser.add_argument('--treename', type=str, default='DataR',\
                        help='Name of the tree containing the data in a .root file')
    return parser

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
        data = np.loadtxt(file_path)
    except ValueError:
        data = np.loadtxt(file_path, usecols=(0, 1))

    if len(data.shape) == 1: # This is the # Energy format
        content = data
    elif data.shape[1] == 2: # This is the # Time, Energy format
        content = data[:,1]
    elif data.shape[1] == 3: # This is the # Time, Energy, Gain format og GammaStream MC2 lists
        content = data[:,1]
    else: # This is an unknown format
        print("Unknown column format for this file. Please check it.")
        sys.exit(1)
    return content

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

    if len(data.shape) == 1: # This is the # Energy format
        content = data
    elif data.shape[1] == 2: # This is the # Time, Energy format
        content = data[:,1]
    elif data.shape[1] == 3: # This is the # Time, Energy, Gain format og GammaStream MC2 lists
        content = data[:,1]
    else: # This is an unknown format
        print("Unknown column format for this file. Please check it.")
        sys.exit(1)
    return content
    
def import_from_root(file_path: str | os.PathLike, treename: str='Data_R') -> np.array:
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
    # ... and returning it	
    return spectrum

def check_file_format(file_path: str | os.PathLike) -> str:
    """Check the file format of a given file.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to the file.

    Returns
    -------
    file_format : str
        The format of the file.
    """
    if file_path.endswith('.txt'):
        return 'txt'
    elif file_path.endswith('.csv'):
        return 'csv'
    elif file_path.endswith('.root'):
        return 'root'
    else:
        raise ValueError("File format not supported.")

def import_spectrum(file_path: str | os.PathLike, **kwargs) -> Tuple[np.array, np.array]:
    if file_path.endswith('.txt'):
        return import_from_txt(file_path)
    elif file_path.endswith('.csv'):
        return import_from_csv(file_path)
    elif file_path.endswith('.root'):
        return import_from_root(file_path, kwargs.get('treename', 'Data_R'))
    else:
        raise ValueError("File format not supported.")

