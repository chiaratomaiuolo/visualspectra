# System libraries first...
import os
import sys
import tkinter as tk
from tkinter import messagebox
# ... then installed libraries...
import matplotlib.pyplot as plt
# ... and eventually local modules.
from spectratools.spectraio import create_parser
from spectratools.spectraplot import SpectraPlotter

def main():
    # Parsing arguments from parser
    parser = create_parser()
    args = parser.parse_args()

    # Filling the file list obtained from parser
    for file in args.filepaths:
        try:
            app = SpectraPlotter(args.filepaths, nbins=args.nbins, treename=args.treename)
            app.mainloop()
        except FileNotFoundError:
            messagebox.showwarning(message=f"File {os.path.basename(file)} not found.")
        except ValueError:
            messagebox.showerror(message=f"File format not supported.")

if __name__ == "__main__":
    main()