""" Utilities for the spectral analysis.
"""

# System libraries first...
import os
# ... then installed libraries...
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
# ... and eventually local modules
import spectratools.spectraio as io_utils

def onselect(xmin, xmax):
    """Callback function to handle the selection of an interval."""
    print(f"Selected interval: {xmin} - {xmax}")
    # You can add additional logic here to handle the selected interval
