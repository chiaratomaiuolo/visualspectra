"""Utilities for GUI creation and spectral plot
"""
# System libraries first...
import os
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
        # Setting the 'current file' for fit purposes
        # to the last file in the list
        if len(self.file_paths) != 0:
            self.current_file = self.file_paths[-1]
            # Importing the data of the current file
            data = io_utils.import_spectrum(self.current_file)
            # Constructing the current histogram and saving it as an attribute.
            # NB: (from Matplotlib documentation) If the data has already been 
            # binned and counted, use bar or stairs to plot the distribution. 
            self.current_spectrum = np.histogram(data, bins=self.nbins)
        self.roi_limits = []
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
        self.add_button = ttk.Button(self, text="Add Spectrum", command=self.add_spectra)
        self.add_button.place(x=10, y=10)

        # Rebin spectrum button
        self.rebin_button = ttk.Button(self, text="Rebin Spectrum", bootstyle='success',\
                                       command=self.rebin_spectra)
        self.rebin_button.place(x=120, y=10)

        # Select current spectrum button
        self.select_button = ttk.Button(self, text="Select current spectrum", bootstyle='default',\
                                        command=self.select_spectrum)
        self.select_button.place(x=240, y=10)
        # Delete opened file button
        self.delete_button = ttk.Button(self, text="Close spectrum", bootstyle='danger',\
                                        command=self.delete_file)
        self.delete_button.place(x=400, y=10)
        # Interval selection button
        # Adding SpanSelector for interval selection
        self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                 handle_props={"alpha":0.5, "facecolor":'red'})
        self.span.set_active(False)  # Initially deactivate the SpanSelector

        # Add interval selection button
        self.interval_button = ttk.Button(self, text="Select ROI", bootstyle='info',\
                                          command=self.toggle_span_selector)
        self.interval_button.place(x=520, y=10)
        # Connect the motion notify event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        # Connect the key press event
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        # Set focus to the canvas to capture key events
        self.canvas.get_tk_widget().focus_set()

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
                    spectrum = io_utils.import_spectrum(file, treename=self.treename)
                    hist = self.ax.hist(spectrum, bins=self.nbins, alpha=0.6,\
                                        label=f'{os.path.basename(file)}')
                    self.histograms[file] = hist[2]  # Save the patches (rectangles) of the histogram
                except FileNotFoundError:
                    Messagebox.show_warning("Warning", f"File {file} not found.")

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if self.file_paths[i] == self.current_file\
                            else 'black' for i in range(len(labels))]
            self.ax.legend(handles, labels, labelcolor=label_colors)

            # Adding the axes labels
            self.ax.set_xlabel("Energy [ADC counts]")
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
                spectrum = io_utils.import_spectrum(file_path, treename=self.treename)
                hist = self.ax.hist(spectrum, bins=self.nbins,\
                                    label=f'{os.path.basename(file_path)}')
                self.histograms[file_path] = hist[2]  # Save the patches of the histogram
            except FileNotFoundError:
                Messagebox.show_warning("Warning", f"File {file_path} not found.")

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if self.file_paths[i] == self.current_file\
                            else 'black' for i in range(len(labels))]
            self.ax.legend(handles, labels, labelcolor=label_colors)

            self.ax.grid(True)
            self.ax.autoscale()  # Autoscale the axes
            self.canvas.draw()

    # -------------- BUTTON DEFINITION FUNCTIONS --------------

    # -------------- ADD SPECTRA BUTTON --------------
    def add_spectra(self):
        file_path = filedialog.askopenfilename(title="Select a file", \
                                               filetypes=[("ROOT files", "*.root"),\
                                                          ("CSV files", "*.csv"),\
                                                          ("TXT files", "*.txt")])
        if file_path:
            self.current_file = file_path
            self.current_spectrum = np.histogram(io_utils.import_spectrum(file_path), bins=self.nbins)
            self.file_paths.append(file_path)
            self.plot_spectra()
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
                self.current_spectrum = np.histogram(io_utils.import_spectrum(selected_file), bins=self.nbins)
                self.plot_spectra()
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
        # Defining the roi_binning
        roi_binning = self.current_spectrum[1][roi_mask]
        popt, pcov = analysis_utils.onselect(self.current_spectrum, xmin, xmax)
        # Tracing the vertical lines defining the ROI
        self.ax.axvline(x=xmin, linestyle='--', linewidth=1, color='red')
        self.ax.axvline(x=xmax, color=plt.gca().lines[-1].get_color(),\
                        linestyle='--', linewidth=1)
        self.ax.annotate(f'{self.roi_limits.index(new_roi)}', xy=(xmin, 0),\
                         xytext=(xmax-(xmax-xmin)*0.5, 100),  fontsize=12)
        # Plotting the fit results on spectrum
        self.ax.plot(roi_binning, analysis_utils.GaussLine(roi_binning, popt))

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


    # -------------- CLOSING PROTOCOL --------------
    def on_closing(self):
        self.quit()
        self.destroy()
