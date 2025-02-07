"""Utilities for GUI creation and spectral plot
"""
# System libraries first...
from datetime import datetime
import os
from pathlib import Path
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
# ... then installed libraries...
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
# ... and eventually local modules
import spectratools.spectraanalysis as analysis_utils
import spectratools.spectraio as io_utils

class SpectraPlotter(ttk.Window):
    def __init__(self, file_paths: list[str | os.PathLike], nbins: int, **kwargs):
        """Class constructor
        """
        super().__init__(themename="flatly")
        # Setting closing protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Loading file paths as class attributes
        self.file_paths = file_paths
        self.nbins = nbins
        self.density = False
        # Setting the 'current file' for fit purposes
        # to the last file in the list
        if len(self.file_paths) != 0:
            self.current_file = self.file_paths[-1]
            # Importing the data of the current file
            data = io_utils.import_spectrum(self.current_file, treename=self.get_treename(self.current_file))
            # Constructing the current histogram and saving it as an attribute.
            # NB: (from Matplotlib documentation) If the data has already been 
            # binned and counted, use bar or stairs to plot the distribution. 
            self.current_spectrum = np.histogram(data, bins=self.nbins)
        # Creating lists for storing the ROIs and their fit results
        self.roi_limits = []
        self.roi_popt = []
        self.roi_dpopt = []
        self.roi_file = [] # tracking the spectrum relative to a specific ROI
        # Creating dictionaries containing calibration points and factors for each file
        # Flag that indicates the x scale 
        self.xscale_unit = 'ADC' # Default unit is ADC when canva is created
        self.calibration_points = {}
        self.calibration_factors = {}
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
            if len(self.file_paths) == 0:
                # If there are no files to plot, just opening a white window
                self.canvas.draw()
                return
            for file in self.file_paths:
                try:
                    spectrum = io_utils.import_spectrum(file, treename=self.get_treename(file))
                    # Checking x axis unit
                    if self.xscale_unit == 'keV':
                        if file in self.calibration_factors:
                            m, q = self.calibration_factors[file]
                            spectrum = analysis_utils.adc_to_kev(spectrum, m, q)
                        else:
                            spectrum = analysis_utils.adc_to_kev(spectrum, 1, 0)
                    if self.density is False:
                        hist = self.ax.hist(spectrum, bins=np.linspace(1,max(spectrum),self.nbins), alpha=0.6,\
                                            label=f'{os.path.basename(file)}')
                        self.histograms[file] = hist[2]  # Save the patches (rectangles) of the histogram
                    else:
                        hist = self.ax.hist(spectrum, bins=np.linspace(1,max(spectrum),self.nbins), alpha=0.6,\
                                            label=f'{os.path.basename(file)}', density=True)
                        self.histograms[file] = hist[2]  # Save the patches (rectangles) of the histogram
                except FileNotFoundError:
                    Messagebox.show_warning("Warning", f"File {file} not found.")

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if self.file_paths[i] == self.current_file\
                            else 'black' for i in range(len(labels))]
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
        else: # If a file path is provided, plot only that file on the current axes
            try:
                if file_path in self.histograms:
                    for patch in self.histograms[file_path]:
                        patch.remove()
                    del self.histograms[file_path]
                spectrum = io_utils.import_spectrum(file_path, treename=self.get_treename(file_path))
                if self.density is False:
                    hist = self.ax.hist(spectrum[10:], bins=self.nbins, alpha=0.6,\
                                        label=f'{os.path.basename(file_path)}')
                    self.histograms[file_path] = hist[2]  # Save the patches of the histogram
                else:
                    hist = self.ax.hist(spectrum[10:], bins=self.nbins, alpha=0.6,\
                                        label=f'{os.path.basename(file_path)}', density=True)
                    self.histograms[file_path] = hist[2]
            except FileNotFoundError:
                Messagebox.show_warning("Warning", f"File {file_path} not found.")

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if self.file_paths[i] == self.current_file\
                            else 'black' for i in range(len(labels))]
            self.ax.legend(handles, labels, labelcolor=label_colors)

            self.ax.grid(True)
            self.ax.set_xlim(10, None)
            #self.ax.autoscale()  # Autoscale the axes
            self.canvas.draw()

    def roi_draw(self, density):
        for i, roi in enumerate(self.roi_limits):
            roi_mask = (self.current_spectrum[1] >= roi[0]) & (self.current_spectrum[1] <= roi[1])
            roi_binning = self.current_spectrum[1][roi_mask]
            popt = self.roi_popt[i]
            self.ax.axvline(x=roi[0], linestyle='--', linewidth=1, color='red')
            self.ax.axvline(x=roi[1], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=1)
            if density is False:
                self.ax.plot(roi_binning, analysis_utils.GaussLine(roi_binning, popt))
                x_annotate = roi[0] + (roi[1]-roi[0])*0.5
                y_annotate = (analysis_utils.GaussLine(roi_binning, popt).max())*0.8
                self.ax.annotate(f'{i}', xy=(roi[0], 0), xytext=(x_annotate, y_annotate), fontsize=12)
            else:
                self.ax.plot(roi_binning, (analysis_utils.GaussLine(roi_binning, popt))/(self.current_spectrum[0].sum()*(roi_binning[1]-roi_binning[0])))
                x_annotate = roi[0] + (roi[1]-roi[0])*0.5
                y_annotate = (analysis_utils.GaussLine(roi_binning, popt).max()/(self.current_spectrum[0].sum()*(roi_binning[1]-roi_binning[0])))*0.8
                self.ax.annotate(f'{i}', xy=(roi[0], 0), xytext=(x_annotate, y_annotate), fontsize=12)
        self.canvas.draw()

    # -------------- BUTTON DEFINITION FUNCTIONS --------------

    # -------------- ADD SPECTRA BUTTON --------------
    def add_spectra(self):
        file_path = filedialog.askopenfilename(title="Select a file", \
                                               filetypes=[("ROOT files", "*.root"),\
                                                          ("CSV files", "*.csv"),\
                                                          ("TXT files", "*.txt")])
        if file_path:
            file_name = os.path.basename(file_path)
            self.current_file = file_path
            self.current_spectrum = np.histogram(io_utils.import_spectrum(file_path, treename=self.get_treename(file_path)), bins=self.nbins)
            self.file_paths.append(file_path)
            self.plot_spectra()
            if self.roi_limits:
                    # If ROis are present, let's replot them
                    self.roi_draw(self.density)
        else:
            Messagebox.show_error("Error", f"File {file_path} not found.")

    # -------------- REBIN SPECTRA BUTTON --------------
    def rebin_spectra(self):
        if not self.file_paths:
            Messagebox.show_warning("Warning", "No files to rebin.")
            return

        # Create a new window for rebinning
        rebin_window = ttk.Toplevel(self)
        rebin_window.title("Rebin Spectrum")
        rebin_window.geometry("300x250")

        ttk.Label(rebin_window, text="Select File:").pack(pady=10)
        file_var = ttk.StringVar(rebin_window)
        file_menu = ttk.Combobox(rebin_window, textvariable=file_var, values=self.file_paths)
        file_menu.pack(pady=10)

        ttk.Label(rebin_window, text="Select Number of Bins:").pack(pady=10)
        bins_var = ttk.IntVar(rebin_window)
        bins_menu = ttk.Combobox(rebin_window, textvariable=bins_var,\
                                values=[128, 256, 512, 1024, 2048, 4096])
        bins_menu.pack(pady=10)

        def apply_rebin():
            selected_file = file_var.get()
            new_bins = bins_var.get()
            if selected_file and new_bins:
                self.nbins = new_bins
                # Remove the old histogram
                if selected_file in self.histograms:
                    for patch in self.histograms[selected_file]:
                        patch.remove()
                    del self.histograms[selected_file]
                # Plot the new histogram
                self.plot_spectra(file_path=selected_file)
                rebin_window.destroy()
            else:
                Messagebox.show_warning("Warning", "Please select a file and number of bins.")

        apply_button = ttk.Button(rebin_window, text="Apply", command=apply_rebin)
        apply_button.pack(pady=10)


    # -------------- NORMALIZE SPECTRA BUTTON --------------
    def normalize_spectra(self):
        if not self.file_paths:
            Messagebox.show_warning("Warning", "No histogram to normalize.")
            return
        if self.density is False:
            self.density = True
            self.plot_spectra()
            # If some ROIs are already defined, replot them
            if self.roi_limits:
                self.roi_draw(self.density)
                self.canvas.draw()
            return
        else:
            self.density = False
            self.plot_spectra()
            # If some ROIs are already defined, replot them
            if self.roi_limits:
                self.roi_draw(self.density)
            return

    # -------------- SELECT SPECTRUM BUTTON --------------
    def select_spectrum(self):
        if not self.file_paths:
            Messagebox.show_warning("Warning", "No files to rebin.")
            return
        # Create a new window for rebinning
        select_window = ttk.Toplevel(self)
        select_window.title("Select Spectrum")
        select_window.geometry("300x250")

        ttk.Label(select_window, text="Select File:").pack(pady=10)
        file_select = ttk.StringVar(select_window)
        file_menu_select = ttk.Combobox(select_window, textvariable=file_select,\
                                        values=self.file_paths)
        file_menu_select.pack(pady=10)

        def apply_selection():
            selected_file = file_select.get()
            if selected_file:
                self.current_file = selected_file
                self.current_spectrum = np.histogram(io_utils.import_spectrum(selected_file, treename=self.get_treename(selected_file)), bins=self.nbins)
                self.plot_spectra()
                # If some ROIs are already defined, replot them
                if self.roi_limits:
                    self.roi_draw(self.density)
                select_window.destroy()
            else:
                Messagebox.show_warning("Warning", "Please select a file.")

        apply_button = ttk.Button(select_window, text="Apply", command=apply_selection)
        apply_button.pack(pady=10)

    # -------------- DELETE FILE BUTTON --------------
    def delete_file(self):
        if not self.file_paths:
            Messagebox.show_warning("Warning", "No file selected.")
            return
        # Create a new window for rebinning
        delete_window = ttk.Toplevel(self)
        delete_window.title("Delete Spectrum")
        delete_window.geometry("300x250")

        ttk.Label(delete_window, text="Select File:").pack(pady=10)
        file_delete = ttk.StringVar(delete_window)
        file_menu_delete = ttk.Combobox(delete_window, textvariable=file_delete,\
                                        values=self.file_paths)
        file_menu_delete.pack(pady=10)

        def apply_deletion():
            selected_file = file_delete.get()
            if selected_file:
                self.file_paths.remove(selected_file)
                if selected_file in self.histograms:
                    for patch in self.histograms[selected_file]:
                        patch.remove()
                    del self.histograms[selected_file]
                self.plot_spectra()
                delete_window.destroy()
            else:
                Messagebox.show_warning("Warning", "Please select a file.")

        apply_button = ttk.Button(delete_window, text="Apply", command=apply_deletion)
        apply_button.pack(pady=10)

    # -------------- INTERVAL SELECTION BUTTON --------------
    def onselect(self, xmin, xmax):
        """Callback function to handle the selection of an interval."""
        # Adding the Tuple containing the ROI limits to the dedicated class attribute
        new_roi = (xmin, xmax)
        self.roi_limits.append(new_roi)
        # Creating the ROI mask for further use
        roi_mask = (self.current_spectrum[1] >= xmin) & (self.current_spectrum[1] <= xmax)
        roi_mask_tmp = roi_mask[:-1]
        # Defining the roi_binning
        roi_binning = self.current_spectrum[1][roi_mask]
        popt, dpopt = analysis_utils.onselect(self.current_spectrum, xmin, xmax,\
                                              density=self.density)
        # Saving fit results
        self.roi_popt.append(popt)
        self.roi_dpopt.append(dpopt)
        self.roi_file.append(self.current_file)
        # Tracing the vertical lines defining the ROI
        self.ax.axvline(x=xmin, linestyle='--', linewidth=1, color='red')
        self.ax.axvline(x=xmax, color=plt.gca().lines[-1].get_color(),\
                        linestyle='--', linewidth=1)
        x_annotate = xmin + (xmax-xmin)*0.5
        y_annotate = (analysis_utils.GaussLine(roi_binning, popt).max())*0.8
        self.ax.annotate(f'{self.roi_limits.index(new_roi)}', xy=(xmin, 0),\
                         xytext=(x_annotate, y_annotate),  fontsize=12)

        # Plotting the fit results on spectrum
        w = np.linspace(min(roi_binning), max(roi_binning), 1000)
        if self.density is False:
            self.ax.plot(w, analysis_utils.GaussLine(w, popt))
        else:
            self.ax.plot(w, (analysis_utils.GaussLine(w, popt))/(self.current_spectrum[0].sum()*(roi_binning[1]-roi_binning[0])))

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
        if not self.roi_limits:
            Messagebox.show_warning("Warning", "No ROI to delete.")
            return
        # Create a new window for ROI selection
        delete_roi_window = ttk.Toplevel(self)
        delete_roi_window.title("Delete ROI")
        delete_roi_window.geometry("300x250")

        ttk.Label(delete_roi_window, text="Select ROI:").pack(pady=10)
        roi_delete = ttk.StringVar(delete_roi_window)
        roi_menu_delete = ttk.Combobox(delete_roi_window, textvariable=roi_delete,\
                                        values=[str(i) for i in range(len(self.roi_limits))] + ['All'])
        roi_menu_delete.pack(pady=10)

        def apply_roideletion():
            selected_roi = roi_delete.get()
            if selected_roi:
                if selected_roi == 'All':
                    self.roi_limits = [] # re-initializing the ROI list
                    self.roi_popt = [] # re-initializing the ROI fit results list
                    self.roi_dpopt = [] # re-initializing the ROI fit errors list
                    self.roi_file = [] # re-initializing the ROI file list
                    # Removing all ROI lines from the plot
                    for line in self.ax.lines[:]:
                        line.remove()
                    # Removing all annotations
                    for annotation in self.ax.texts:
                        annotation.remove()
                    self.canvas.draw()
                    delete_roi_window.destroy()
                else:
                    selected_roi = int(selected_roi)
                    roi_limits = self.roi_limits[selected_roi]
                    # Removing the ROI from the list of ROIs
                    self.roi_popt.pop(selected_roi)
                    self.roi_limits.remove(roi_limits)
                    self.roi_dpopt.pop(selected_roi)
                    self.roi_file.pop(selected_roi)
                    # Removing the ROI lines from the plot
                    for line in self.ax.lines:
                        if (line.get_linestyle() == '--' and line.get_xdata()[0] in roi_limits)\
                            or (line.get_xdata()[0] >= roi_limits[0] and line.get_xdata()[0] <= roi_limits[1]):
                            line.remove()
                    # Removing the corresponding annotations
                    for annotation in self.ax.texts:
                        annotation.remove()
                    # Update the indices of the remaining ROIs and their annotations
                    for i, roi, popt in enumerate(zip(self.roi_limits, self.roi_popt)):
                        roi_mask = (self.current_spectrum[1] >= roi[0]) & (self.current_spectrum[1] <= roi[1])
                        roi_binning = self.current_spectrum[1][roi_mask]
                        x_annotate = roi[0] + (roi[1]-roi[0])*0.5
                        y_annotate = (analysis_utils.GaussLine(roi_binning, popt).max()/(self.current_spectrum[0].sum()*(roi_binning[1]-roi_binning[0])))*0.8
                        self.ax.annotate(f'{i}', xy=(roi[0], 0), xytext=(x_annotate, y_annotate), fontsize=12)

                    self.canvas.draw()
                    delete_roi_window.destroy()
            else:
                Messagebox.show_warning("Warning", "Please select a ROI.")

        apply_button = ttk.Button(delete_roi_window, text="Apply", command=apply_roideletion)
        apply_button.pack(pady=10)

    # -------------- SAVE RESULTS BUTTON --------------
    def save_results(self):
        if not self.roi_limits:
            Messagebox.show_warning("Warning", "No ROI to save.")
            return
        else:
            # Chiedi all'utente di inserire il nome del file
            file_name = self.ask_file_name()
            if not file_name:
                Messagebox.show_warning("Warning", "File name cannot be empty.")
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
                file.write(f'# Source file: {self.current_file}\n')
                file.write(f'# Date of creation of this .txt file: {date_string}\n')
                file.write('# ROI ID    xmin    xmax    mu  dmu sigma   dsigma     res FWHM\n')
                for i, (roi, fitresults, dfitresults) in enumerate(zip(self.roi_limits, self.roi_popt, self.roi_dpopt)):
                    file.write(f'{i}    {roi[0]}    {roi[1]}    {fitresults[3]}    {dfitresults[3]}    {fitresults[4]}    {dfitresults[4]}     {(fitresults[4]/fitresults[3])*2.355}\n')
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
        spectrum_file = ttk.StringVar(value=self.current_file if self.file_paths else "")
        spectrum_menu = ttk.Combobox(dialog, textvariable=spectrum_file, values=self.file_paths)
        spectrum_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        tree = ttk.Treeview(dialog, columns=("Bin", "Energy"), show="headings", bootstyle='info')
        tree.heading("Bin", text="Bin Number")
        tree.heading("Energy", text="Energy [keV]")
        if self.calibration_points.get(spectrum_file.get()):
            for bin_number, energy in self.calibration_points[self.current_file]:
                tree.insert("", "end", values=(bin_number, energy), tags=("row",))
        tree.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Configure row height
        style = ttk.Style()
        style.configure("Treeview", rowheight=30)

        # Menu a tendina per selezionare l'ID del ROI
        ttk.Label(dialog, text="ROI ID:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        roi_id = ttk.StringVar()
        roi_menu = ttk.Combobox(dialog, textvariable=roi_id, values=[str(i) for i in range(len(self.roi_limits))])
        roi_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        def apply_roi():
            selected_roi = roi_id.get()
            if selected_roi:
                Messagebox.show_info("Info", f"Selected ROI ID: {selected_roi}")
            else:
                Messagebox.show_warning("Warning", "Please select a ROI ID.")

        ttk.Button(dialog, text="Apply", command=apply_roi).grid(row=2, column=2, padx=10, pady=5, sticky="ew")

        def add_row():
            tree.insert("", "end", values=("", ""), tags=("row",))

        def on_calibrate():
            selected_file = spectrum_file.get()
            if not selected_file:
                Messagebox.show_warning("Warning", "Please select a spectrum file.")
                return
            calibration_points = [] #List filled with tuples (bin_number, energy)
            for row in tree.get_children():
                bin_number, energy = tree.item(row)["values"]
                if type(bin_number) == int and self.is_float(energy):
                    calibration_points.append((int(bin_number), float(energy)))
                else:
                    Messagebox.show_warning("Warning", "Please enter valid numbers for bin and energy.")
                    return
            self.save_calibration(selected_file, calibration_points)
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
        ttk.Button(dialog, text="Calibrate", command=on_calibrate, bootstyle='info')\
            .grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        dialog.grid_rowconfigure(1, weight=1)
        dialog.grid_columnconfigure(1, weight=1)
        dialog.wait_window(dialog)

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def save_calibration(self, selected_file, calibration_points):
        """Apply the calibration to the spectrum."""
        # Fitting a line to calibration points
        m, q = analysis_utils.calibration_fit(calibration_points)
        # Save the result into the calibration_points dictionary
        self.calibration_points[selected_file] = calibration_points
        self.calibration_factors[selected_file] = (m, q)

    # -------------- CONVERT UNITS BUTTON --------------

    def apply_conversion(self):
        # Converting from ADC to keV
        if self.xscale_unit == 'ADC':
            self.xscale_unit = 'keV'
            self.plot_spectra()
        elif self.xscale_unit == 'keV':
            self.xscale_unit = 'ADC'
            self.plot_spectra()
    
    # -------------- CLEAR ALL BUTTON --------------
    def clear_all(self):
        if not self.file_paths:
            Messagebox.show_warning("Warning", "No files to clear.")
            return
        # Clear all histograms from the plot
        for file in self.file_paths:
            if file in self.histograms:
                for patch in self.histograms[file]:
                    patch.remove()
                del self.histograms[file]
        self.file_paths = []
        self.current_file = None
        self.current_spectrum = None
        self.roi_limits = []
        self.roi_popt = []
        self.roi_dpopt = []
        self.roi_file = []
        self.xscale_unit = 'ADC' # Default unit is ADC when canva is created
        self.calibration_points = {}
        self.calibration_factors = {}
        self.ax.clear()
        self.canvas.draw()

    # -------------- CLOSING PROTOCOL --------------
    def on_closing(self):
        self.quit()
        self.destroy()
