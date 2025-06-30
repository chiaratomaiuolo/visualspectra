# visualspectra
Source code repository of a software tool for spectra visualization. It permits to open files contaning a list of events, construct the relative histogram and visualize it through a GUI implemented with `ttkbootstrap`, hosting a `matplotlib` figure.  

In addition to spectra creation, the tool permits to perform basic analyses:
- ROI selection and its fit with a Gaussian model;
- Energy calibration through manual input of calibration points or selection of ROI centroids
- ...

### Supported filetypes
The supported file types (by now) for the events list are:
- `.root`
- `.csv`
- `.txt`
- `.n42`


