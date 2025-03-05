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
from scipy.optimize import curve_fit
from uncertainties import ufloat
# ... and eventually local modules
import visualspectra.spectraio as io_utils

class Roi():
    """ Class containing all the interestig characteristics of a ROI.
    """
    def __init__(self, roi_limits: list[float, float]):
        self.limits = roi_limits
        @property
        def limits(self):
            return self._limits
        @limits.setter
        def limits(self, limits):
            self._limits = limits
        @property
        def id(self):
            return self._id
        @id.setter
        def id(self, id):
            self._id = id
        @property
        def roi_popt(self):
            return self._roi_popt
        @roi_popt.setter
        def roi_popt(self, roi_popt):
            self._roi_popt = roi_popt
        @property
        def roi_dpopt(self):
            return self._roi_dpopt
        @roi_dpopt.setter
        def roi_dpopt(self, roi_dpopt):
            self._roi_dpopt = roi_dpopt

# ---------------------- Fit functions ----------------------
def line(x: np.ndarray, pars: np.ndarray) -> float:
    """ Linear function in ROOT-like format.
    """
    return pars[0]*x + pars[1]

def linear(x, m, q) -> float:
    """ Linear function in Python-like format.
    """
    return m*x + q

def Gauss(x, pars: np.ndarray) -> float:
    """ Gaussian function in ROOT-like format.
    """
    return pars[0]*np.exp(-(x-pars[1])**2/(2*pars[2]**2))

def GaussLine(x, pars: np.ndarray) -> float:
    """ Gaussian + Linear background function in ROOT-like format.
    """
    return (pars[0]*x + pars[1]) + pars[2]/pars[4]*np.exp(-(x-pars[3])**2/(2.*pars[4]**2))

# ---------------------- Initial parameters computation ----------------------
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

def roi_fit(spectrum: np.array, roi_min: float, roi_max: float, density: bool=False) -> Tuple[np.array, np.array]:
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
    # Defining the ROOT model (Gaussian + Linear background)
    gaussline = root.TF1("gaussline", "[0]*x + [1] + [2]/[4]*exp(-(x-[3])**2/(2.*[4]**2))", roi_bins[0], roi_bins[-1])
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
    print(f'Number of events inside the ROI: {roi_content.sum()}')
    print(f'Number of events from the fit: {popt[2]}')

    # Returning fit parameters and their errors
    return np.array(popt), np.array(dpopt)

def onselect(spectrum, xmin, xmax, density: bool=False):
    """Callback function to handle the selection of an interval.
       Once the ROI has been selected, a fit with a GaussLine model is performed inside.
    """
    # Performing the fit inside the ROI
    popt, dpopt = roi_fit(spectrum, xmin, xmax, density=density)
    # Printing the parameters on terminal...
    par_names = ['m', 'q', 'N', 'mu', 'sigma']
    for name, par, dpar in zip(par_names, popt, dpopt):
        print(f'{name} = {par} +/- {dpar}')
    # ... eventually returning them to the SpectraPlotter class.
    return popt, dpopt


def calibration_fit(calibration_points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """ Function for the fitting of the calibration points.
        The fit function is a linear function.
    """
    # Unpacking the calibration points
    E_adc = np.array([point[0] for point in calibration_points])
    E_kev = np.array([point[1] for point in calibration_points])

    popt, pcov = curve_fit(linear, E_adc, E_kev)
    m, q = popt
    print(m, q)
    return m, q

def adc_to_kev(adc: float, m: float, q: float) -> float:
    """ Function for the conversion of an ADC value to a keV value.
    """
    return m*adc + q

def kev_to_adc(kev: float, m: float, q: float) -> float:
    """ Function for the conversion of an keV value to an ADC value.
    """
    return (kev - q)/m

