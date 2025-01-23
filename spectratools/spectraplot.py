# System libraries first...
import os
import tkinter as tk
from tkinter import filedialog, messagebox
# ... then installed libraries...
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
# ... and eventually local modules
import spectratools.spectraio as io_utils

class SpectraPlotter(tk.Tk):
    def __init__(self, file_paths: list[str | os.PathLike], **kwargs):
        super().__init__()
        # Setting closing protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Loading file paths as class attributes
        self.file_paths = file_paths
        # Saving possible additional arguments
        if kwargs:
            self.kwargs_keys = []
            self.process_kwargs(kwargs)
        #self.nbins = nbins
        #self.treename = treename
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
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Adding the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        #self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_spectra(self.file_paths)
        self.add_button = tk.Button(self, text="Add Spectra", command=self.add_spectra)
        self.add_button.pack(side=tk.BOTTOM)

    def plot_spectra(self, file_paths, ):
        self.ax.set(xlabel='Energy', ylabel='Counts')
        for file in file_paths:
            try:
                spectrum = io_utils.import_spectrum(file, treename=self.treename)
                self.ax.hist(spectrum, bins=self.nbins, label=f'{os.path.basename(file)}')
                #plt.hist(bins[:-1], bins=bins, weights=content, edgecolor='black')
            except FileNotFoundError as e:
                messagebox.showwarning("Warning", f"File {file} not found.")
        self.ax.legend()
        self.ax.autoscale()  # Autoscale the axes
        self.canvas.draw()

    def add_spectra(self):
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("CSV files", "*.csv"), ("TXT files", "*.txt"), ("ROOT files", "*.root")])
        if file_path:
            self.plot_spectra([file_path])
        else:
            messagebox.showerror("Error", f"File {file_path} not found.")

    def on_closing(self):
        self.quit()
        self.destroy()
