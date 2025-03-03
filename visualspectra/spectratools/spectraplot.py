"""Utilities for GUI creation and spectral plot
"""
# System libraries first...
from datetime import datetime
import os
from pathlib import Path
from tkinter import filedialog
import ttkbootstrap as ttk
from typing import List
from ttkbootstrap.dialogs import Messagebox
# ... then installed libraries...
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
# ... and eventually local modules
import spectratools.spectraanalysis as analysis_utils
from  spectratools.spectraanalysis import Roi
import spectratools.spectraio as io_utils


def rescale_spectrum(evt_list: List | np.array, new_nbins: int) -> np.array:
    #starting_bins = max(evt_list) + 1
    starting_bins = 16384
    print(starting_bins)
    refactor = starting_bins / new_nbins
    print(refactor)
    evt_list = evt_list/refactor #rounding to ints the refactored list
    return evt_list


class SpectraPlotter(ttk.Window):
    def __init__(self, file_paths: list[str | os.PathLike], nbins: int, **kwargs):
        """Class constructor
        """
        super().__init__(themename="flatly")
        # Setting closing protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Creating a dictionary of dictionaries containing the infos about the
        # opened files
        self.nbins = nbins # It is needed for having a standard when adding new files
        self.opened_spectra = {}
        if file_paths:
            for file in file_paths:
                # Creating histogram
                data =  io_utils.import_spectrum(file, treename=self.get_treename(file))
                bins = np.arange(0, self.nbins+1, 1)
                # Storing the rescaled histogram referred to the chosen nbins number
                histogram = np.histogram(rescale_spectrum(data, self.nbins), bins)
                self.opened_spectra[file] = {
                    'data': data,
                    'nbins': self.nbins,
                    'histogram': histogram,
                    'rois' : [],
                    'fine_gain': 1.0,
                    'calibration_points': [],
                    'calibration_factors': []
                }
        # Setting the 'current file' for fit purposes
        self.current_file = None
        # to the last file in the list
        if len(self.opened_spectra) != 0:
            self.current_file = file_paths[-1]
            # Importing the data of the current file
            # Constructing the current histogram and saving it as an attribute.
            # NB: (from Matplotlib documentation) If the data has already been 
            # binned and counted, use bar or stairs to plot the distribution. 
            self.current_spectrum = self.opened_spectra[file_paths[-1]]['histogram']
        # Creating dictionaries containing calibration points and factors for each file
        # Flag that indicates the x scale 
        self.density = False
        self.xscale_unit = 'ADC' # Default unit is ADC when canva is created
        self.current_roi_number = None # Counter for the ROIs
        # Line to follow the cursor
        self.cursor_line = None
        # Saving possible additional arguments
        if kwargs:
            self.kwargs_keys = []
            self.process_kwargs(kwargs)
        # Dictionary to keep track of plotted histograms
        self.histograms = {}
        # Initializing the spectrum window
        self.initialize_ui()

    def process_kwargs(self, kwargs):
        """ Keyword arguments handling for SpectraPlotter class when passed
            to an istance constructor.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.kwargs_keys.append(key)

    def initialize_ui(self):
        """ User Interface initialization for SpectraPlotter class.
        """
        # Canvas definition
        self.title("Spectra Plotter")
        self.geometry("1200x800")
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=ttk.BOTH, expand=True)

        # Adding the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=ttk.TOP, fill=ttk.X)
        # Creating the Matplotlib figure inside the canva
        self.plot_spectra()
        # -------------- BUTTONS DEFINITION --------------
        # Add spectrum button
        self.add_button = ttk.Button(self, text="Add Spectrum",  bootstyle='success',\
                                     command=self.add_spectra)
        self.add_button.place(x=10, y=10)

        # Rebin spectrum button
        self.rebin_button = ttk.Button(self, text="Rebin Spectrum", bootstyle='success',\
                                       command=self.rebin_spectra)
        self.rebin_button.place(x=120, y=10)

        # Normalize spectra button
        self.normalize_button = ttk.Button(self, text="Normalize/Un-normalize Spectrum", bootstyle='success',\
                                             command=self.normalize_spectra)
        self.normalize_button.place(x=520, y=10)

        # Select current spectrum button
        self.select_button = ttk.Button(self, text="Select current spectrum", bootstyle='success',\
                                        command=self.select_spectrum)
        self.select_button.place(x=240, y=10)
        # Delete opened file button
        self.delete_button = ttk.Button(self, text="Close spectrum", bootstyle='success',\
                                        command=self.delete_file)
        self.delete_button.place(x=400, y=10)
        # Interval selection button
        # Adding SpanSelector for interval selection
        self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                 handle_props={"alpha":0.5, "facecolor":'red'})
        self.span.set_active(False)  # Initially deactivate the SpanSelector

        # Add interval selection button
        self.interval_button = ttk.Button(self, text="Select ROI", bootstyle='primary',\
                                          command=self.toggle_span_selector)
        self.interval_button.place(x=740, y=10)
        # Connect the motion notify event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        # Connect the key press event
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        # Set focus to the canvas to capture key events
        self.canvas.get_tk_widget().focus_set()

        # Delete ROI(s) button
        self.deleteroi_button = ttk.Button(self, text="Delete ROI(s)", bootstyle='primary',\
                                        command=self.delete_roi)
        self.deleteroi_button.place(x=820, y=10)

        # Save ROI fit results button
        self.save_button = ttk.Button(self, text="Save ROI fit results", bootstyle='primary',\
                                      command=self.save_results)
        self.save_button.place(x=920, y=10)
        # Button for inserting calibration points and calibrating the spectrum
        self.calibrate_button = ttk.Button(self, text="Calibrate spectrum", bootstyle='info', command=self.calibrate_spectrum)
        self.calibrate_button.place(x=10, y=50)

        # Button for conversion from/to ADC/keV
        self.convert_button = ttk.Button(self, text="ADC/keV conversion", bootstyle='info', command=self.apply_conversion)
        self.convert_button.place(x=150, y=50)

        # Button for applying a fine gain to the spectrum
        self.finegain_button = ttk.Button(self, text="Apply fine gain", bootstyle='info', command=self.fine_gain)
        self.finegain_button.place(x=300, y=50)



        # Button for clearing all
        self.clear_button = ttk.Button(self, text="Clear all", bootstyle='danger', command=self.clear_all)
        self.clear_button.place(x=1120, y=10)

    @staticmethod
    def get_treename(filepath: str | Path) -> str:
        filename = os.path.basename(filepath)
        if filename.startswith('DataR'):
            return 'Data_R'
        elif filename.startswith('DataF'):
            return 'Data_F'
        elif filename.startswith('Data'):
            return 'Data'
        else:
            return None

    def plot_spectra(self, file_path: str = None):
        """ Method plotting the spectra in the User Interface canva.
        """
        if not file_path: # If a file path is not provided, plot (or re-plot) all opened files
            self.ax.clear()
            if len(self.opened_spectra) == 0:
                # If there are no files to plot, just opening a white window
                self.canvas.draw()
                return
            for file_name, file_content in self.opened_spectra.items():
                try:
                    nbins = file_content['nbins']
                    bins = np.arange(0, nbins+1, 1)
                    spectrum = rescale_spectrum(file_content['data'], nbins)
                    # Checking x axis unit
                    if self.xscale_unit == 'keV':
                        if file_content['calibration_factors']: # If calibration factors are present
                            m, q = file_content['calibration_factors']
                            spectrum = analysis_utils.adc_to_kev(spectrum, m, q)
                        else:
                            spectrum = analysis_utils.adc_to_kev(spectrum, 1, 0)
                    if self.density is False:
                        hist = self.ax.hist(spectrum*file_content['fine_gain'], bins=bins, alpha=0.6,\
                                            label=f'{os.path.basename(file_name)}')
                        self.histograms[file_name] = hist[2]  # Save the patches (rectangles) of the histogram
                    else:
                        hist = self.ax.hist(spectrum*file_content['fine_gain'], bins=bins, alpha=0.6,\
                                            label=f'{os.path.basename(file_name)}', density=True)
                        self.histograms[file_name] = hist[2]  # Save the patches (rectangles) of the histogram
                except FileNotFoundError:
                    Messagebox.show_warning(f"File {file_name} not found.", "Warning")

                # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if key == self.current_file\
                            else 'black' for key in self.opened_spectra.keys()]
            self.ax.legend(handles, labels, labelcolor=label_colors)

            # Adding the axes labels
            if self.xscale_unit == 'ADC':
                self.ax.set_xlabel("Energy [ADC counts]")
            if self.xscale_unit == 'keV':
                self.ax.set_xlabel("Energy [keV]")
            self.ax.set_ylabel("Counts")
            self.ax.grid(True)
            self.ax.autoscale()  # Autoscale the axes
            self.canvas.draw()
        else: # If a file path is provided, re-plot only that file on the current axes
            try:
                if file_path in self.histograms:
                    for patch in self.histograms[file_path]:
                        patch.remove()
                    del self.histograms[file_path]
            except FileNotFoundError:
                Messagebox.show_warning("Warning", f"File {file_path} not found.")
                return
            nbins = self.opened_spectra[file_path]['nbins']
            bins = np.arange(0, nbins+1, 1)
            spectrum = rescale_spectrum(self.opened_spectra[file_path]['data'], nbins)
            if self.density is False:
                hist = self.ax.hist(spectrum*self.opened_spectra[file_path]['fine_gain'], bins=bins, alpha=0.6,\
                                    label=f'{os.path.basename(file_path)}')
                self.histograms[file_path] = hist[2]  # Save the patches of the histogram
            else:
                hist = self.ax.hist(spectrum*self.opened_spectra[file_path]['fine_gain'], bins=bins,\
                                    alpha=0.6, label=f'{os.path.basename(file_path)}', density=True)
                self.histograms[file_path] = hist[2]

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if key == self.current_file\
                            else 'black' for key in self.opened_spectra.keys()]
            self.ax.legend(handles, labels, labelcolor=label_colors)

            self.ax.grid(True)
            self.ax.set_xlim(10, None)
            self.ax.autoscale()  # Autoscale the axes
            self.canvas.draw()

    def roi_draw(self):
        for file_name, file_content in self.opened_spectra.items():
            # Checking if there are ROIs relative to the spectrum
            if file_content['rois']:
                # If so, let's plot them
                hist = file_content['histogram']
                rois = file_content['rois']
                for roi in rois:
                    roi_id = roi.id
                    roi_limits = roi.limits
                    roi_popt = roi.roi_popt
                    roi_mask = (hist[1] >= roi_limits[0]) & (hist[1] <= roi_limits[1])
                    roi_binning = hist[1][roi_mask] #Filtered bins content
                    if self.xscale_unit == 'keV' and file_content['calibration_factors']:
                        # Need to rescale the ROI limits
                        m, q = file_content['calibration_factors']
                        roi = (analysis_utils.adc_to_kev(np.array([roi_limits[0]]), m, q),\
                            analysis_utils.adc_to_kev(np.array([roi_limits[1]]), m, q))
                        roi_binning = analysis_utils.adc_to_kev(hist[1][roi_mask], m, q)
                        # Rescaling fit parameters
                        # Need to doc how I have derived the parameters
                        roi_popt = [roi_popt[0]/m, roi_popt[1]-(q/m)*roi_popt[0], roi_popt[2]*m, m*(roi_popt[3]+q), roi_popt[4]*m]
                    self.ax.axvline(x=analysis_utils.adc_to_kev(roi_limits[0], m, q), linestyle='--', linewidth=1, color='red')
                    self.ax.axvline(x=analysis_utils.adc_to_kev(roi_limits[1], m, q), color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=1)
                    if self.density is False:
                        w = np.linspace(min(roi_binning), max(roi_binning), 1000)
                        self.ax.plot(w, analysis_utils.GaussLine(w, roi_popt))
                        x_annotate = analysis_utils.adc_to_kev(roi_limits[0], m, q) +\
                            (analysis_utils.adc_to_kev(roi_limits[1], m, q) - analysis_utils.adc_to_kev(roi_limits[0], m, q))*0.5
                        y_annotate = (analysis_utils.GaussLine(roi_binning, roi_popt).max())*0.8
                        self.ax.annotate(f'{roi_id}', xy=(roi_limits[0], 0), xytext=(x_annotate, y_annotate), fontsize=12)
                    else:
                        self.ax.plot(roi_binning, (analysis_utils.GaussLine(roi_binning, roi_popt))/(hist[0].sum()*(roi_binning[1]-roi_binning[0])))
                        x_annotate = roi_limits[0] + (roi_limits[1]-roi_limits[0])*0.5
                        y_annotate = (analysis_utils.GaussLine(roi_binning, roi_popt).max()/(hist[0].sum()*(roi_binning[1]-roi_binning[0])))*0.8
                        self.ax.annotate(f'{roi_id}', xy=(roi_limits[0], 0), xytext=(x_annotate, y_annotate), fontsize=12)
        # Finally drawing the canva
        self.canvas.draw()

    # -------------- BUTTON DEFINITION FUNCTIONS --------------

    # -------------- ADD SPECTRA BUTTON --------------
    def add_spectra(self):
        file_path = filedialog.askopenfilename(title="Select a file", \
                                               filetypes=[("ROOT files", "*.root"),\
                                                          ("CSV files", "*.csv"),\
                                                          ("TXT files", "*.txt")])
        if not file_path:
            Messagebox.show_error(f"No valid file provided.", "Error")
        if file_path not in self.opened_spectra.keys() and file_path is not None:
            # New file, need to create a new dictionary entry
            data =  io_utils.import_spectrum(file_path, treename=self.get_treename(file_path))
            bins = np.arange(0, self.nbins+1, 1)
            # Storing the rescaled histogram referred to the chosen nbins number
            histogram = np.histogram(rescale_spectrum(data, self.nbins), bins)
            self.opened_spectra[file_path] = {
                'data': data,
                'nbins': self.nbins,
                'histogram': histogram,
                'fine_gain': 1.0,
                'rois' : [],
                'calibration_points': [],
                'calibration_factors': []
            }
            # Changing the current file to the new one
            self.current_file = file_path
            self.current_spectrum = self.opened_spectra[file_path]['histogram']
            # Replotting the spectra in the canva with the new file
            self.plot_spectra()
            if self.current_roi_number is not None:
                    # If ROis are present, let's replot them
                    self.roi_draw()
        else:
            Messagebox.show_error(f"File already in the canva.", "Error")

    # -------------- REBIN SPECTRA BUTTON --------------
    def rebin_spectra(self):
        if not self.opened_spectra.keys():
            Messagebox.show_warning("No files to rebin.", "Warning")
            return

        # Create a new window for rebinning
        rebin_window = ttk.Toplevel(self)
        rebin_window.title("Rebin Spectrum")
        rebin_window.geometry("300x250")

        ttk.Label(rebin_window, text="Select File:").pack(pady=10)
        file_var = ttk.StringVar(value=self.current_file if self.opened_spectra.keys() else "")
        file_menu = ttk.Combobox(rebin_window, textvariable=file_var,\
                                 values=list(self.opened_spectra.keys()))
        file_menu.pack(pady=10)

        ttk.Label(rebin_window, text="Select Number of Bins:").pack(pady=10)
        file_var.get()
        bins_var = ttk.IntVar(value=self.opened_spectra[file_var.get()]['nbins'])
        bins_menu = ttk.Combobox(rebin_window, textvariable=bins_var,\
                                 values=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
        bins_menu.pack(pady=10)

        def apply_rebin():
            selected_file = file_var.get()
            new_nbins = bins_var.get()
            if selected_file and new_nbins:
                # Changing the binning to the selected file...
                self.opened_spectra[selected_file]['nbins'] = new_nbins
                new_bins = np.arange(0, new_nbins+1, 1)
                rescaled_data = rescale_spectrum(self.opened_spectra[selected_file]['data'], new_nbins)
                # ... re-creating the histogram with the new binning
                self.opened_spectra[selected_file]['histogram'] = \
                    np.histogram(rescaled_data, bins=new_bins)
                # If the re-binned histogram is the current one, updating it
                if selected_file == self.current_file:
                    self.current_spectrum = self.opened_spectra[selected_file]['histogram']
                # Remove the old histogram
                if selected_file in self.histograms.keys():
                    for patch in self.histograms[selected_file]:
                        patch.remove()
                    del self.histograms[selected_file]
                # Plot the new histogram
                self.plot_spectra(file_path=selected_file)
                # eventually, if ROIs are associated to the spectrum, re-computing the fit params
                # and replotting on the new histogram
                if self.opened_spectra[selected_file]['rois']:
                    rois = self.opened_spectra[selected_file]['rois']
                    for roi in rois:
                        for line in self.ax.lines:
                            if (line.get_linestyle() == '--' and line.get_xdata()[0] in roi.limits)\
                                or (line.get_xdata()[0] >= roi.limits[0] and line.get_xdata()[0] <= roi.limits[1]):
                                line.remove()
                        # Re-fitting the ROIs without incrementing the ROI number
                        self.onselect(roi.limits[0], roi.limits[1], increase=False)
                rebin_window.destroy()
            else:
                Messagebox.show_warning("Please select a file and number of bins.", "Warning")

        apply_button = ttk.Button(rebin_window, text="Apply", command=apply_rebin)
        apply_button.pack(pady=10)


    # -------------- NORMALIZE SPECTRA BUTTON --------------
    def normalize_spectra(self):
        if not self.opened_spectra.keys():
            Messagebox.show_warning("No histogram to normalize.", "Warning")
            return
        if self.density is False:
            self.density = True
            self.plot_spectra()
            # If some ROIs are already defined, replot them
            if self.current_roi_number is not None:
                # In this case, at least a ROI does exists
                self.roi_draw()
            return
        else:
            self.density = False
            self.plot_spectra()
            # If some ROIs are already defined, replot them
            if self.current_roi_number is not None:
                self.roi_draw()
            return

    # -------------- SELECT SPECTRUM BUTTON --------------
    def select_spectrum(self):
        if not self.opened_spectra.keys():
            Messagebox.show_warning("No files to select.", "Warning")
            return
        # Create a new window for rebinning
        select_window = ttk.Toplevel(self)
        select_window.title("Select Spectrum")
        select_window.geometry("300x250")

        ttk.Label(select_window, text="Select File:").pack(pady=10)
        file_select = ttk.StringVar(select_window)
        file_menu_select = ttk.Combobox(select_window, textvariable=file_select,\
                                        values=list(self.opened_spectra.keys()))
        file_menu_select.pack(pady=10)

        def apply_selection():
            selected_file = file_select.get()
            if selected_file:
                self.current_file = selected_file
                self.current_spectrum = self.opened_spectra[selected_file]['histogram']
                self.plot_spectra()
                # If some ROIs are already defined, replot them
                if self.current_roi_number is not None:
                    self.roi_draw()
                select_window.destroy()
            else:
                Messagebox.show_warning("Please select a file.", "Warning")

        apply_button = ttk.Button(select_window, text="Apply", command=apply_selection)
        apply_button.pack(pady=10)

    # -------------- DELETE FILE BUTTON --------------
    def delete_file(self):
        if not self.opened_spectra.keys():
            Messagebox.show_warning("No opened file.", "Warning")
            return
        # Create a new window for rebinning
        delete_window = ttk.Toplevel(self)
        delete_window.title("Delete Spectrum")
        delete_window.geometry("300x250")

        ttk.Label(delete_window, text="Select File:").pack(pady=10)
        file_delete = ttk.StringVar(delete_window)
        file_menu_delete = ttk.Combobox(delete_window, textvariable=file_delete,\
                                        values=list(self.opened_spectra.keys()))
        file_menu_delete.pack(pady=10)

        def apply_deletion():
            selected_file = file_delete.get()
            if selected_file:
                # Removing the selected file from the opened_spectra dictionary
                del self.opened_spectra[selected_file]
                self.current_file = list(self.opened_spectra.keys())[-1] if self.opened_spectra.keys() else None
                if selected_file in self.histograms:
                    # Cancelling the histogram from the canva
                    for patch in self.histograms[selected_file]:
                        patch.remove()
                    del self.histograms[selected_file]
                self.plot_spectra()
                if self.current_roi_number is not None:
                    # If ROIs are present
                    self.roi_draw()
                delete_window.destroy()
            else:
                Messagebox.show_warning("Please select a file.", "Warning")

        apply_button = ttk.Button(delete_window, text="Apply", command=apply_deletion)
        apply_button.pack(pady=10)

    # -------------- INTERVAL SELECTION BUTTON --------------
    def onselect(self, xmin, xmax):
        """Callback function to handle the selection of an interval."""
        # Adding the Tuple containing the ROI limits to the dedicated class attribute
        new_roi = (xmin, xmax)
        # Selecting the rois dictionary for the current file
        hist = self.opened_spectra[self.current_file]['histogram']
        rois = self.opened_spectra[self.current_file]['rois'] # This is a list of Roi objects
        # Appending the new ROI to the list of ROIs corresponding to the file
        # Verifica se new_roi è presente in uno dei limits degli oggetti Roi
        roi_index = next((i for i, roi in enumerate(rois) if roi.limits == new_roi), None)
        if roi_index is not None:
            # Searching for roi index in order to update the fit results
            # Creating the ROI mask for further use
            roi_mask = (hist[1] >= xmin) & (hist[1] <= xmax)
            # Defining the roi_binning
            roi_binning = hist[1][roi_mask]
            popt, dpopt = analysis_utils.onselect(hist, xmin, xmax, density=self.density)
            # Saving 'new' fit results
            rois[roi_index].roi_popt = popt
            rois[roi_index].roi_dpopt = dpopt
        else:
            # New ROI, need to be fitted and then appended to the list
            roi = Roi(new_roi)
            # Creating the ROI mask for further use
            roi_mask = (hist[1] >= xmin) & (hist[1] <= xmax)
            # Defining the roi_binning
            roi_binning = hist[1][roi_mask]
            popt, dpopt = analysis_utils.onselect(hist, xmin, xmax,\
                                                density=self.density)
            # Saving fit results
            roi.roi_popt = popt
            roi.roi_dpopt = dpopt
            if self.current_roi_number is None:
                # First roi in the canva
                self.current_roi_number = 0
            roi.id = self.current_roi_number
            rois.append(roi)
            # Incrementing the global ROI number
        # Tracing the vertical lines defining the ROI
        self.ax.axvline(x=xmin, linestyle='--', linewidth=1, color='red')
        self.ax.axvline(x=xmax, color=plt.gca().lines[-1].get_color(),\
                        linestyle='--', linewidth=1)
        x_annotate = xmin + (xmax-xmin)*0.5
        if self.density is False:
            y_annotate = (analysis_utils.GaussLine(roi_binning, popt).max())*0.8
            self.ax.annotate(f'{self.current_roi_number}', xy=(xmin, 0),\
                        xytext=(x_annotate, y_annotate),  fontsize=12)
        else:
            y_annotate = (analysis_utils.GaussLine(roi_binning, popt).max()/(hist[0].sum()*(roi_binning[1]-roi_binning[0])))*0.8
            self.ax.annotate(f'{self.current_roi_number}', xy=(xmin, 0),\
                        xytext=(x_annotate, y_annotate),  fontsize=12)
        # Eventually updating the current roi number flag
        self.current_roi_number += 1

        # Plotting the fit results on spectrum
        w = np.linspace(min(roi_binning), max(roi_binning), 1000)
        if self.density is False:
            self.ax.plot(w, analysis_utils.GaussLine(w, popt))
        else:
            self.ax.plot(w, (analysis_utils.GaussLine(w, popt))/(hist[0].sum()*(roi_binning[1]-roi_binning[0])))

        self.canvas.draw()

    def toggle_span_selector(self):
        """Toggle the activation of the SpanSelector."""
        self.span.set_active(not self.span.active)
        if self.span.active:
            self.cursor_line = self.ax.axvline(color='gray', linestyle='--')
        else:
            if self.cursor_line:
                self.cursor_line.remove()
                self.cursor_line = None
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Handle the mouse move event to update the cursor line."""
        if self.span.active and event.inaxes == self.ax:
            if self.cursor_line:
                self.cursor_line.set_xdata([event.xdata])
            else:
                self.cursor_line = self.ax.axvline(x=event.xdata, color='gray',\
                                                   linestyle='--')
            self.canvas.draw()

    def on_key_press(self, event):
        """Handle the key press event to deactivate the SpanSelector on Esc key press."""
        if event.key == 'escape' and self.span.active:
            self.toggle_span_selector()

    # -------------- DELETE ROI BUTTON --------------
    def delete_roi(self):
        if not self.current_roi_number:
            Messagebox.show_warning("No ROI to delete.", "Warning")
            return
        # Create a new window for ROI selection
        delete_roi_window = ttk.Toplevel(self)
        delete_roi_window.title("Delete ROI")
        delete_roi_window.geometry("300x250")

        ttk.Label(delete_roi_window, text="Select ROI:").pack(pady=10)
        roi_delete = ttk.StringVar(delete_roi_window)
        roi_menu_delete = ttk.Combobox(delete_roi_window, textvariable=roi_delete,\
                                        values=[str(i) for i in range(self.current_roi_number)] + ['All'])
        roi_menu_delete.pack(pady=10)

        def update_roi_menu_delete():
            roi_menu_delete['values'] = [str(roi.id) for roi in self.opened_spectra[self.current_file]['rois']] + ['All']

        update_roi_menu_delete()

        def apply_roideletion():
            selected_roi = roi_delete.get()
            if selected_roi:
                if selected_roi == 'All':
                    for spectrum in self.opened_spectra.values():
                        spectrum['rois'] = []
                    # Removing all ROI lines from the plot
                    for line in self.ax.lines[:]:
                        line.remove()
                    # Removing all annotations
                    for annotation in self.ax.texts:
                        annotation.remove()
                    self.canvas.draw()
                else:
                    # A specific ROI ID has been chosen
                    selected_roi = int(selected_roi)
                    # Removing the corresponding annotations
                    for annotation in self.ax.texts:
                        annotation.remove()
                    for spectrum in self.opened_spectra.values():
                        rois = spectrum['rois']
                        roi_index = next((i for i, roi in enumerate(rois) if roi.id == selected_roi), None)
                        # Search for the ROI to delete
                        rois = spectrum['rois']
                        if roi_index is not None:
                            # Removing the ROI lines for the selected ROI from the plot
                            for line in self.ax.lines:
                                if (line.get_linestyle() == '--' and line.get_xdata()[0] in rois[roi_index].limits)\
                                    or (line.get_xdata()[0] >= rois[roi_index].limits[0] and line.get_xdata()[0] <= rois[roi_index].limits[1]):
                                    line.remove()
                            # Eventually deleting the ROI from the list
                            rois.pop(roi_index)
                        # Update the indices of the remaining ROIs and their annotations
                        for roi in rois:
                            roi_mask = (spectrum['histogram'][1] >= roi.limits[0])\
                                     & (spectrum['histogram'][1] <= roi.limits[1])
                            roi_binning = spectrum['histogram'][1][roi_mask]
                            x_annotate = roi.limits[0] + (roi.limits[1]-roi.limits[0])*0.5
                            if self.density is False:
                                y_annotate = analysis_utils.GaussLine(roi_binning, roi.roi_popt).max()*0.8
                            else:
                                y_annotate = (analysis_utils.GaussLine(roi_binning,\
                                roi.popt).max()/(spectrum['histogram'][0].sum()\
                                *(roi_binning[1]-roi_binning[0])))*0.8
                            if roi.id < selected_roi:
                                # Nothing changes
                                self.ax.annotate(f'{roi.id}', xy=(roi.limits[0], 0),\
                                xytext=(x_annotate, y_annotate), fontsize=12)
                            else:
                                # Decrementing the ROI number
                                roi.id = roi.id - 1
                                # Decrementing the total number of ROIs
                                self.current_roi_number -= 1
                                # Updating the annotation
                                self.ax.annotate(f'{roi.id}', xy=(roi.limits[0], 0),\
                                xytext=(x_annotate, y_annotate), fontsize=12)
                    # Plotting the canva
                    self.canvas.draw()
                update_roi_menu_delete()
            else:
                Messagebox.show_warning("Please select a ROI.", "Warning")

        apply_button = ttk.Button(delete_roi_window, text="Apply", command=apply_roideletion)
        apply_button.pack(pady=10)

    # -------------- SAVE RESULTS BUTTON --------------
    def save_results(self):
        if self.current_roi_number is None:
            Messagebox.show_warning("No ROI to save", "Warning")
            return
        else:
            # Chiedi all'utente di inserire il nome del file
            file_name = self.ask_file_name()
            if not file_name:
                Messagebox.show_warning("File name cannot be empty", "Warning")
                return

            # Ottieni la data e l'ora corrente
            now = datetime.now()
            # Formatta la data e l'ora come stringa
            date_string = now.strftime("%Y-%m-%d %H:%M:%S")
            # Aggiungi un timestamp al nome del file per evitare sovrascritture
            file_name = f"{file_name}.txt"
            # Ottieni il percorso del file
            file_path = ((Path(__file__).parent).parent).parent / 'fitresults' / file_name
            # Crea la directory se non esiste
            #file_path.parent.mkdir(parents=True, exist_ok=True)
            # Scrivi i risultati nel file
            with open(file_path, 'w') as file:
                # Writing header
                file.write(f'# Source file(s): {self.opened_spectra.keys()}\n')
                file.write(f'# Date of creation of this .txt file: {date_string}\n')
                file.write('# ROI ID    xmin    xmax    mu  dmu sigma   dsigma     res FWHM\n')
                for spectra in self.opened_spectra.values():
                    for roi in spectra['rois']:
                        # Selecting roi limits and fit results
                        roi_lims = roi.limits
                        roi_id = roi.id
                        fitresults = roi.roi_popt
                        dfitresults = roi.roi_dpopt
                        file.write(f'{roi_id} {roi_lims[0]} {roi_lims[1]} {fitresults[3]} {dfitresults[3]} {fitresults[4]} {dfitresults[4]} {(fitresults[4]/fitresults[3])*2.355}\n')
            Messagebox.ok(f"{file_path} file created", "Save ROI(s) fit results")

    def ask_file_name(self):
        """Ask the user for a file name using a custom dialog."""
        dialog = ttk.Toplevel(self)
        dialog.title("Enter File Name")
        dialog.geometry("300x150")

        ttk.Label(dialog, text="Enter the name of the file:").pack(pady=10)
        file_name_var = ttk.StringVar()
        file_name_entry = ttk.Entry(dialog, textvariable=file_name_var)
        file_name_entry.pack(pady=10)

        def on_ok():
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        dialog.wait_window(dialog)
        return file_name_var.get()
    
    # -------------- CALIBRATE SPECTRUM BUTTON ----------------
    def calibrate_spectrum(self):
        """Open a dialog to input bin number and corresponding energy for calibration."""
        dialog = ttk.Toplevel(self)
        dialog.title("Spectrum calibration")
        dialog.geometry("500x600")

        # Menu a tendina per selezionare il nome del file dello spettro da calibrare
        ttk.Label(dialog, text="Select Spectrum:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        spectrum_file = ttk.StringVar(value=self.current_file if self.opened_spectra.keys() else "")
        spectrum_menu = ttk.Combobox(dialog, textvariable=spectrum_file, values=list(self.opened_spectra.keys()))
        spectrum_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        tree = ttk.Treeview(dialog, columns=("Bin", "Energy"), show="headings", bootstyle='info')
        tree.heading("Bin", text="Bin Number")
        tree.heading("Energy", text="Energy [keV]")
        # Need an if that looks if there are no rows []
        tree.insert("", "end", values=(0, 0), tags=("row",))
        if self.opened_spectra.get(spectrum_file.get())['calibration_points']:
            calibration_points = self.opened_spectra.get(spectrum_file.get())['calibration_points']
            for bin_number, energy in calibration_points:
                tree.insert("", "end", values=(bin_number, energy), tags=("row",))
        tree.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Configure row height
        style = ttk.Style()
        style.configure("Treeview", rowheight=30)

        # ROI ID selection window
        ttk.Label(dialog, text="ROI ID:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        roi_id = ttk.StringVar()
        roi_menu = ttk.Combobox(dialog, textvariable=roi_id)
        roi_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        def update_roi_menu(*args):
            selected_file = spectrum_file.get()
            if selected_file in self.opened_spectra:
                current_rois = self.opened_spectra[selected_file]['rois']
                roi_menu['values'] = [str(roi.id) for roi in current_rois]
                # Clear the treeview
                for row in tree.get_children():
                    tree.delete(row)
                # Insert calibration points for the selected file
                calibration_points = self.opened_spectra[selected_file]['calibration_points']
                for bin_number, energy in calibration_points:
                    tree.insert("", "end", values=(bin_number, energy), tags=("row",))
            else:
                roi_menu['values'] = []
                # Clear the treeview
                for row in tree.get_children():
                    tree.delete(row)

        spectrum_file.trace_add('write', update_roi_menu)
        update_roi_menu()

        def apply_roi():
            selected_file = spectrum_file.get()
            selected_roi = int(roi_id.get())
            if selected_roi is not None:
                current_spectrum = self.opened_spectra.get(selected_file)
                rois = current_spectrum['rois']
                roi_index = next((i for i, roi in enumerate(rois) if roi.id == selected_roi), None)
                popt = rois[roi_index].roi_popt
                # We want to use the ROI centroid as the energy value
                tree.insert("", "end", values=(f"{popt[3]}", ""), tags=("row",))
            else:
                Messagebox.show_warning("Please select a ROI ID.", "Warning")

        ttk.Button(dialog, text="Apply", command=apply_roi).grid(row=2, column=2, padx=10, pady=5, sticky="ew")

        def add_row():
            tree.insert("", "end", values=("", ""), tags=("row",))
        
        def delete_row():
            selected_item = tree.selection()
            if selected_item:
                tree.delete(selected_item)

        def on_calibrate():
            selected_file = spectrum_file.get()
            spectrum = self.opened_spectra.get(selected_file)
            if not selected_file:
                Messagebox.show_warning("Please select a spectrum file", "Warning")
                return
            for row in tree.get_children():
                bin_number, energy = tree.item(row)["values"]
                if self.is_float(bin_number) and self.is_float(energy):
                    spectrum['calibration_points'].append((float(bin_number), float(energy))) #List filled with tuples (bin_number, energy)
                else:
                    Messagebox.show_warning("Please enter valid numbers for bin and energy", "Warning")
                    return
            self.apply_calibration(selected_file, spectrum['calibration_points'])
        def on_double_click(event):
            item = tree.selection()[0]
            column = tree.identify_column(event.x)
            column_index = int(column[1:]) - 1
            x, y, width, height = tree.bbox(item, column)
            value = tree.item(item, 'values')[column_index]
            entry = ttk.Entry(dialog)
            entry.place(x=x + tree.winfo_rootx() - dialog.winfo_rootx(),\
                        y=y + tree.winfo_rooty() - dialog.winfo_rooty(),\
                        width=width, height=height)
            entry.insert(0, value)
            entry.focus()

            def on_focus_out(event):
                tree.set(item, column, entry.get())
                entry.destroy()

            def on_return(event):
                tree.set(item, column, entry.get())
                entry.destroy()

            entry.bind("<FocusOut>", on_focus_out)
            entry.bind("<Return>", on_return)

        tree.bind("<Double-1>", on_double_click)

        ttk.Button(dialog, text="Add Row", command=add_row, bootstyle='info')\
            .grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(dialog, text="Delete Row", command=delete_row, bootstyle='danger')\
            .grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        ttk.Button(dialog, text="Calibrate", command=on_calibrate, bootstyle='info')\
            .grid(row=3, column=2, padx=10, pady=5, sticky="ew")

        dialog.grid_rowconfigure(1, weight=1)
        dialog.grid_columnconfigure(1, weight=1)
        dialog.wait_window(dialog)

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def apply_calibration(self, selected_file, calibration_points):
        """Apply the calibration to the spectrum."""
        # Fitting a line to calibration points
        m, q = analysis_utils.calibration_fit(calibration_points)
        # Save the result into the calibration_factors dictionary
        spectrum = self.opened_spectra[selected_file]
        spectrum['calibration_factors'] = (m, q)

    # -------------- CONVERT UNITS BUTTON --------------

    def apply_conversion(self):
        # Converting from ADC to keV
        if self.xscale_unit == 'ADC':
            self.xscale_unit = 'keV'
            self.plot_spectra()
            if self.current_roi_number is not None:
                self.roi_draw()
        elif self.xscale_unit == 'keV':
            self.xscale_unit = 'ADC'
            self.plot_spectra()
            if self.current_roi_number is not None:
                self.roi_draw()

    # -------------- FINE GAIN BUTTON --------------
    def fine_gain(self):
        if not self.opened_spectra.keys():
            Messagebox.show_warning("No files to select.", "Warning")
            return
        # Create a new window for rebinning
        select_window = ttk.Toplevel(self)
        select_window.title("Select Spectrum")
        select_window.geometry("300x250")

        ttk.Label(select_window, text="Select File:").pack(pady=10)
        file_select = ttk.StringVar(value=self.current_file if self.opened_spectra.keys() else "")
        file_menu_select = ttk.Combobox(select_window, textvariable=file_select,\
                                        values=list(self.opened_spectra.keys()))
        file_menu_select.pack(pady=10)

        ttk.Label(select_window, text="Enter Fine Gain:").pack(pady=10)
        fine_gain_var = ttk.StringVar(value=self.opened_spectra[file_select.get()]['fine_gain'])
        fine_gain_entry = ttk.Entry(select_window, textvariable=fine_gain_var)
        fine_gain_entry.pack(pady=10)

        def apply_fine_gain():
            selected_file = file_select.get()
            fine_gain_value = float(fine_gain_var.get())
            if selected_file:
                # Changing the fine gain of the selected hist...
                self.opened_spectra[selected_file]['fine_gain'] = fine_gain_value
                self.opened_spectra[selected_file]['histogram'] =\
                np.histogram(rescale_spectrum(self.opened_spectra[selected_file]['data'],\
                            self.nbins)*fine_gain_value, bins=np.arange(0, self.nbins+1, 1))
                
                # And re-plotting it
                self.plot_spectra()
                # If some ROIs are already defined, replot them
                if self.current_roi_number is not None:
                    self.roi_draw()
                select_window.destroy()
            else:
                Messagebox.show_warning("Please select a file.", "Warning")

        apply_button = ttk.Button(select_window, text="Apply", command=apply_fine_gain)
        apply_button.pack(pady=10)
    
    # -------------- CLEAR ALL BUTTON --------------
    def clear_all(self):
        if not self.opened_spectra.keys():
            Messagebox.show_warning("No files to clear", "Warning")
            return
        # Clearing the filled class instance attributes
        self.nbins = 1024
        self.opened_spectra = {}
        self.density = False
        self.xscale_unit = 'ADC' # Default unit is ADC when canva is created
        self.current_roi_number = None # Counter for the ROIs
        # Line to follow the cursor
        self.cursor_line = None
        self.ax.clear()
        self.canvas.draw()

    # -------------- CLOSING PROTOCOL --------------
    def on_closing(self):
        self.quit()
        self.destroy()
