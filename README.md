# DicomSeriesManager

## Description

```DicomSeriesManager``` is a package for reading and displaying DICOM series built on top of pydicom and matplotlib.

Its main features are:
- a class hierarchy (based on ```BaseSeries```) that defines series and dynamic series and allows easy access to their metadata and pixel values,
- a class ```DicomDirContent``` that enables a triage of DICOM images into series and a save of the image paths to allow an easy loading of any given series,
- a function ```show``` to allow an easy display of any image or slice,
- various functions to get slices and slice properties for any of the three main orientations (axial, coronal, sagittal).


## Dependencies

Python packages:
- numpy
- matplotlib
- pydicom

## Content

- series.py  
  - Classes to define DICOM series
    - ```BaseSeries```
    - |> ```Series```
    - |> ```DynSeries```  
  - Function to generate instances of series
    - ```series_factory```
- reader.py
  - Class to read DICOM files from a directory and sort them into series
    - ```DicomDirContent```
- display.py
  - Classes defining objects to be drawn on images
    - ```Marker```
    - ```Line```
  - Function to display an image or slice
    - ```show```
- reorientation.py
  - Function to reorient pixel data
    - ```reorient_from_axial```
  - Functions to reorient the coordinates of a point
    - ```get_reorientated_slice_index```
    - ```get_reorientated_marker_indices```
  - Functions to reorient volume shapes
    - ```get_reorientated_n_slices```
    - ```get_reorientated_im_shape```
  - Functions to reorient physical dimensions
    - ```get_reorientated_FOV```
    - ```get_reorientated_PS```
- utils.py
  - Functions to get slice geometry
    - ```get_IPP```
    - ```get_IOP```
    - ```get_normal```
    - ```get_PS```
    - ```get_slice_spacing```
  - Function to check series of slices
    - ```check_slice_spacing_consistency```
  - Functions to get transformation matrices
    - ```get_point_to_position_matrix```
    - ```get_point_to_position_matrix_with_depth```
  - Functions to get limit values
    - ```get_window_limits```
    - ```get_slice_limits```
