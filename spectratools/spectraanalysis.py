""" Utilities for the spectral analysis.
"""
# System libraries first...
import os
from typing import Tuple, List
# ... then installed libraries...
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
import ROOT as root
from uncertainties import ufloat
# ... and eventually local modules
import spectratools.spectraio as io_utils

# ---------------------- Fit functions ----------------------
def line(x: np.ndarray, pars: np.ndarray) -> float:
    return pars[0]*x + pars[1]

def Gauss(x, pars: np.ndarray) -> float:
    return pars[0]*np.exp(-(x-pars[1])**2/(2*pars[2]**2))

def GaussLine(x, pars: np.ndarray) -> float:
    return (pars[0]*x + pars[1]) + pars[2]/pars[4]*np.exp(-(x-pars[3])**2/(2.*pars[4]**2))


def init_computation(content: np.array, bins: np.array) -> List[float]:
    """ Function for the computation of the initial parameters of a Gauss + linear 
        BKG fit inside a ROI.
        For the linear part, taking the first and last point of the ROI:
            - m = (y[-1]-y[0])/(x[-1]-x[0]);
            - q = y[0] - m*x[0].
        For the Gaussian part:
            - N = integral of the ROI;
            - mu = centroid of the ROI;
            - sigma = std deviation of the ROI.
    """
    # Computing initial params
    # ------ Gaussian part ------
    N = content.sum()
    print(f'N: {N}')
    mu = np.mean(bins)
    print(f'mu: {mu}')
    sigma = np.std(bins)
    print(f'sigma: {sigma}')
    # ------ Linear part ------
    m = (content[-1]-content[0])/(bins[-1]-bins[0])
    print(f'm: {m}')
    q = content[0] - m*bins[0]
    print(f'q: {q}')
    
    return m, q, N, mu, sigma

def roi_fit(spectrum: np.array, roi_min: float, roi_max: float) -> Tuple[np.array, np.array]:
    """ Function for the fitting of a spectrum inside a ROI.
        The fit function is Gaussian + Linear BKG.
    """
    # Unpacking the spectrum
    bins = spectrum[1]
    content = spectrum[0]
    # Selecting the ROI
    bins = bins[:-1]
    roi_bins = bins[(bins >= roi_min) & (bins <= roi_max)]
    roi_bins = np.array(roi_bins, dtype='float64')
    roi_content = content[(bins >= roi_min) & (bins <= roi_max)]
    roi_content = np.array(roi_content, dtype='float64')
    # Creating a ROOT RDataFrame from the numpy arrays
    # Creation of the TGraph
    graph = root.TGraph(len(roi_bins), roi_bins, roi_content)
    print('Debug print. TGraph created.')
    # Defining the ROOT model (Gaussian + Linear background)
    gaussline = root.TF1("gaussline", "[0]*x + [1] + [2]/[4]*exp(-(x-[3])**2/(2.*[4]**2))", roi_bins[0], roi_bins[-1])
    print('Debug print, TF1 created.')
    # Computing the initial parameters and setting them to the model
    init = init_computation(roi_content, roi_bins)
    gaussline.SetParameters(*init)
    gaussline.SetParNames("m", "q", "N", "mu", "sigma")
    gaussline.Print()
    # Performing the fit - creating the graph with the data and fitting
    fitresults = graph.Fit(gaussline, "S")
    popt = np.array(fitresults.Parameters())
    # Saving the parameters errors
    dpopt = []
    for i in range(len(popt)):
        dpopt.append(fitresults.Error(i))
    # Print the fit results
    fitresults.Print()

    # Returning fit parameters and their errors
    return np.array(popt), np.array(dpopt)


def onselect(spectrum, xmin, xmax):
    """Callback function to handle the selection of an interval.
       Once the ROI has been selected, a fit with a GaussLine model is performed inside.
    """
    # Performing the fit inside the ROI
    popt, dpopt = roi_fit(spectrum, xmin, xmax)
    # Printing the parameters on terminal...
    par_names = ['m', 'q', 'N', 'mu', 'sigma']
    for name, par, dpar in zip(par_names, popt, dpopt):
        print(f'{name} = {par} +/- {dpar}')
    # ... eventually returning them to the SpectraPlotter class.
    return popt, dpopt

