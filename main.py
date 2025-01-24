# System libraries first...
import os
import sys
from ttkbootstrap.dialogs import Messagebox
# ... then installed libraries...
# ... and eventually local modules.
from spectratools.spectraio import create_parser, check_file_format
from spectratools.spectraplot import SpectraPlotter

def main():
    # Parsing arguments from parser
    parser = create_parser()
    args = parser.parse_args()

    # Checking file format
    if not args.filepaths:
        filepaths = []
        app = SpectraPlotter(filepaths, nbins=args.nbins, treename=args.treename)
        app.mainloop()
    else:
        for file in args.filepaths:
            opened_files = []
            if not check_file_format(file):
                Messagebox.show_error(message=f"{file} format not supported.")
                continue
            else:
                opened_files.append(file)
            if not opened_files:
                Messagebox.show_error(message="No valid file to show.")
                sys.exit(1)

        # Filling the file list obtained from parser
        for file in args.filepaths:
            try:
                app = SpectraPlotter(opened_files, nbins=args.nbins, treename=args.treename)
                app.mainloop()
            except FileNotFoundError:
                Messagebox.show_warning(message=f"File {os.path.basename(file)} not found.")
            except ValueError:
                Messagebox.show_error(message=f"File format not supported.")

if __name__ == "__main__":
    main()