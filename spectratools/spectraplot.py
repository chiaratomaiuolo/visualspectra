# System libraries first...
import os
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
# ... then installed libraries...
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
# ... and eventually local modules
import spectratools.spectraio as io_utils

class SpectraPlotter(ttk.Window):
    def __init__(self, file_paths: list[str | os.PathLike], **kwargs):
        super().__init__(themename="flatly")
        # Setting closing protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Loading file paths as class attributes
        self.file_paths = file_paths
        # Setting the 'current file' for fit purposes
        # to the last file in the list
        if len(self.file_paths) != 0:
            self.current_file = self.file_paths[-1]
        # Saving possible additional arguments
        if kwargs:
            self.kwargs_keys = []
            self.process_kwargs(kwargs)
        # Dictionary to keep track of plotted histograms
        self.histograms = {}
        # Initializing the spectrum window
        self.initialize_ui()
    
    def process_kwargs(self, kwargs):
        for key, value in kwargs.items():
                setattr(self, key, value)
                self.kwargs_keys.append(key)

    def initialize_ui(self):
        self.title("Spectra Plotter")
        self.geometry("1200x800")
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=ttk.BOTH, expand=True)

        # Adding the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=ttk.TOP, fill=ttk.X)

        self.plot_spectra()
        self.add_button = ttk.Button(self, text="Add Spectrum", command=self.add_spectra)
        self.add_button.place(x=10, y=10)

        self.rebin_button = ttk.Button(self, text="Rebin Spectrum", bootstyle='success', command=self.rebin_spectra)
        self.rebin_button.place(x=120, y=10)

        self.select_button = ttk.Button(self, text="Select current spectrum", bootstyle='default', command=self.select_spectrum)
        self.select_button.place(x=240, y=10)

    def plot_spectra(self, file_path: str = None):
        if not file_path: # If a file path is not provided, plot (or re-plot) all opened files
            self.ax.clear()
            if len(self.file_paths) == 0:
                # If there are no files to plot, just opening a white window
                self.canvas.draw()
                return
            for file in self.file_paths:
                try:
                    spectrum = io_utils.import_spectrum(file, treename=self.treename)
                    hist = self.ax.hist(spectrum, bins=self.nbins, alpha=0.6, label=f'{os.path.basename(file)}')
                    self.histograms[file] = hist[2]  # Save the patches (rectangles) of the histogram
                except FileNotFoundError as e:
                    Messagebox.show_warning("Warning", f"File {file} not found.")

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if self.file_paths[i] == self.current_file else 'black' for i in range(len(labels))]
            self.ax.legend(handles, labels, labelcolor=label_colors)
            
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
                hist = self.ax.hist(spectrum, bins=self.nbins, label=f'{os.path.basename(file_path)}')
                self.histograms[file_path] = hist[2]  # Save the patches (rectangles) of the histogram
            except FileNotFoundError as e:
                Messagebox.show_warning("Warning", f"File {file_path} not found.")

            # Update legend with custom colors
            handles, labels = self.ax.get_legend_handles_labels()
            label_colors = ['red' if self.file_paths[i] == self.current_file else 'black' for i in range(len(labels))]
            self.ax.legend(handles, labels, labelcolor=label_colors)
            
            self.ax.grid(True)
            self.ax.autoscale()  # Autoscale the axes
            self.canvas.draw()

    def add_spectra(self):
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("ROOT files", "*.root"), ("CSV files", "*.csv"), ("TXT files", "*.txt")])
        if file_path:
            self.current_file = file_path
            self.file_paths.append(file_path)
            self.plot_spectra()
        else:
            Messagebox.show_error("Error", f"File {file_path} not found.")

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
        bins_menu = ttk.Combobox(rebin_window, textvariable=bins_var, values=[128, 256, 512, 1024, 2048, 4096])
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
        file_menu_select = ttk.Combobox(select_window, textvariable=file_select, values=self.file_paths)
        file_menu_select.pack(pady=10)

        def apply_selection():
            selected_file = file_select.get()
            if selected_file:
                self.current_file = selected_file
                self.plot_spectra()
                select_window.destroy()
            else:
                Messagebox.show_warning("Warning", "Please select a file.")
                
        apply_button = ttk.Button(select_window, text="Apply", command=apply_selection)
        apply_button.pack(pady=10)

    def on_closing(self):
        self.quit()
        self.destroy()
