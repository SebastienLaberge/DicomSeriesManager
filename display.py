"""
Module for displaying DICOM series
"""

from matplotlib.colors import Normalize
from matplotlib.pyplot import subplots
import numpy as np
from pydicom import Dataset

from .reorientation import get_reoriented_PS, reorient_from_axial
from .series import Series, DynSeries
from .utils import get_window_limits, get_slice_limits, get_PS


class Marker:
    """
    Marker to be displayed on an image

    Example: Marker(i, j, s=35, marker='o', c='r', edgecolors='b')
    """

    def __init__(self, i, j, **params):

        self.i = i
        self.j = j
        self.params = params

    def plot(self, ax, offset_ij):

        marker_i = _round_to_int(self.i + offset_ij[0])
        marker_j = _round_to_int(self.j + offset_ij[1])

        ax.scatter(marker_j, marker_i, **self.params)


class Line:
    """
    Line to be displayed on an image

    Example: Line(start_ij, stop_ij, color='r', linestyle='-')
    """

    def __init__(self, start_ij, stop_ij, **params):

        self.start_ij = start_ij
        self.stop_ij = stop_ij
        self.params = params

    def plot(self, ax, offset_ij):

        start_i = _round_to_int(self.start_ij[0] + offset_ij[0])
        start_j = _round_to_int(self.start_ij[1] + offset_ij[1])

        stop_i = _round_to_int(self.stop_ij[0] + offset_ij[0])
        stop_j = _round_to_int(self.stop_ij[1] + offset_ij[1])

        line_i = [start_i, stop_i]
        line_j = [start_j, stop_j]

        ax.plot(line_j, line_i, **self.params)


def show(image_data, ax=None, *,
         frame=None, ind=None, orientation='Axial',
         window=None, FOV=None, pixel_spacing=None,
         seg=None, alpha=0.8,
         markers=None, lines=None,
         axis_labels=None, **kwargs):
    """
    -- Parameter image_data: Input data to be displayed

    Can be one of:
    1) A dynamic series as an instance of class DynSeries
    2) A dynamic series as a numpy 4D array (like output of
       DynSeries.goc_stack)
    3) A series as an instance of class Series
    4) A series as a numpy 3D array (like output of
       Series.goc_stack)
    5) An image as an instance of class pydicom.Dataset
    6) An image as a 2D array
    Warning: Input images are assumed to have and axial
             orientation

    -- ax: Object of type Axes where to plot the image

    If absent, a new figure is created containing that Axes
    object

    -- frame: Index of time frame to be displayed

    Required for cases 1 & 2

    -- ind: Index of slice to be displayed according to
            selected orientation

    Required for cases 1, 2, 3 & 4

    -- orientation: Orientation of the image

    Allowed values (case-insensitive):
        'Axial', 'Coronal', 'Sagittal'
    Reorientation is only supported for 3D/4D images (cases 1
    to 4) and for 3D segmentations

    -- window: Display window for the image displayed

    Can be one of:
    A) None (default), in which case the window corresponds to
       the range of pixel values of the slice
    B) A manual window as an iterable object containing two
       elements: [WindowCenter, WindowWidth]
    C) For cases 1, 3 & 5 only, the single index of a window
       specified in DICOM metadata

    Notes on manual window (A):
    For cases 1, 3 & 5:
        Specified on rescaled pixel values
    For cases 2, 4 & 6:
        Specified directly on the pixel values of the array

    -- FOV: Field of view to be displayed

    Format: [width, height, x_offset, y_offset]
        width and height can be None to represent the maximum
        allowed value
    An offset of [0, 0] represents a FOV centered on the center
    of the whole image.
    If pixel spacing is not available:
        Intrinsic units (pixels)
    If pixel spacing is available:
        Real world units (millimeters)

    -- pixel_spacing: Size of a pixel as [height, width]

    If pixel_spacing is available, it determines the displayed
    pixel aspect ratio and affects the interpretation of the
    FOV value

    For cases 1, 3 & 5, the pixel spacing is read from DICOM
    metadata and does not need to be provided. If provided
    anyway, the parameter value overrides the one read from
    DICOM.

    -- seg: Segmentations to display

    Either a single segmentation or a list of segmentations to
    display
    Each segmentation is in turn either a mask array or a tuple
    of size 2 containing an array followed by a RGB triplet for
    the display color.

    The display color defaults to yellow if absent.

    -- alpha: Transparency level of segmentation

    -- markers: List of instances of Marker class to be
                displayed on the image

    -- lines: List of instances of Line class to be displayed
              on the image

    -- axis_labels
    """

    # Get relevant data according to input case
    # dataset: pydicom dataset if available (cases 1, 3 & 5) or
    #          None otherwise
    # image: 2D numpy array giving pixel values to be displayed
    # pixel_spacing_read: Pixel dimensions if dataset available
    #                     (cases 1, 3 & 5), None otherwise
    # rescaled: Boolean that is true if rescaling has been done
    #           on the raw pixel values
    dataset, image, pixel_spacing_read, rescaled = \
        _get_data(image_data, frame, ind, orientation)

    # Override pixel_spacing if provided as input
    if pixel_spacing is not None:
        pixel_spacing_read = pixel_spacing

    # Window: Get range of values for display window and
    #         normalization object
    v_min, v_max = \
        get_window_limits(window, dataset, image, rescaled)
    norm = Normalize(v_min, v_max)

    # If FOV available, get range of pixel indices in both
    # dimensions and crop image accordingly
    if FOV is not None:

        x_range, y_range = get_slice_limits(
            FOV, image.shape, pixel_spacing_read)

        image = image[y_range[0]:y_range[1]+1,
                      x_range[0]:x_range[1]+1]
    else:
        x_range = [0, image.shape[1]]
        y_range = [0, image.shape[0]]

    # Get segmentation slice corresponding to displayed image
    if seg is not None:

        if not isinstance(seg, list):
            seg = [seg]

        def init_seg(single_seg):

            if isinstance(single_seg, tuple):
                mask, color = single_seg
            else:
                # Default color: yellow
                mask = single_seg
                color = [255, 255, 0]

            if frame is not None and mask.ndim >= 4:
                mask = mask[frame]

            if ind is not None and mask.ndim >= 3:
                mask = \
                    reorient_from_axial(
                        mask,
                        orientation,
                        ind)

            if FOV is not None:
                mask = mask[y_range[0]:y_range[1]+1,
                            x_range[0]:x_range[1]+1]

            if mask.shape != image.shape:
                raise ValueError(
                    "Segmentation has incompatible dimensions "
                    "with image")

            return mask, color

        seg = list(map(init_seg, seg))

    # Get pixel aspect ratio (height/width)
    pixel_aspect_ratio = \
        pixel_spacing_read[0]/pixel_spacing_read[1] \
        if pixel_spacing_read is not None \
        else 1.0

    # Get axes
    if ax is None:
        _, ax = subplots()
    else:
        ax.clear()

    # Display image
    axim = ax.imshow(
        image,
        'gray',
        norm,
        aspect=pixel_aspect_ratio,
        **kwargs)

    # Add axis labels
    if axis_labels is None:
        ax.set_axis_off()
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    # Display overlaid segmentation
    if seg is not None:

        for single_seg in seg:

            mask, color = single_seg

            binary_mask = mask != 0
            value = max(0, min(255, round(255*alpha)))
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[:, :, 0][binary_mask] = color[0]
            rgba[:, :, 1][binary_mask] = color[1]
            rgba[:, :, 2][binary_mask] = color[2]
            rgba[:, :, 3][binary_mask] = value

            ax.imshow(rgba, aspect=pixel_aspect_ratio, **kwargs)

    if markers is not None or lines is not None:

        offset_ij = \
            [-y_range[0], -x_range[0]] \
            if FOV is not None \
            else None

        if markers is not None:

            for marker in markers:

                marker.plot(ax, offset_ij)

        if lines is not None:

            for line in lines:

                line.plot(ax, offset_ij)

    return axim


def _get_data(image_data, frame, ind, orientation):

    # Determine case

    # Case 1
    is_dyn_series = isinstance(image_data, DynSeries)
    # Case 3
    is_series = isinstance(image_data, Series)
    # Case 5
    is_dataset = isinstance(image_data, Dataset)

    is_array = isinstance(image_data, np.ndarray)
    is_4d_array = is_array and image_data.ndim == 4  # Case 2
    is_3d_array = is_array and image_data.ndim == 3  # Case 4
    is_2d_array = is_array and image_data.ndim == 2  # Case 6

    # Make orientation parameter case-insensitive
    reoriented = \
        orientation.capitalize() in ['Sagittal', 'Coronal']

    if is_dyn_series or is_4d_array:
        # Cases 1 & 2: Dynamic series

        if frame is None:
            raise ValueError("frame must be provided for "
                             "dynamic series")

        if ind is None:
            raise ValueError("ind must be provided for dynamic "
                             "series")

        if is_dyn_series:
            # Case 1
            if not reoriented:
                # For performance, no need to generate stack,
                # but pixel values aren't rescaled
                dataset = image_data.frames[frame][ind]
                image = dataset.pixel_array
                pixel_spacing = get_PS(dataset)
                rescaled = False
            else:
                # Used for windowing
                dataset = image_data.frames[frame][0]

                image = \
                    reorient_from_axial(
                        image_data.goc_stack()[frame],
                        orientation,
                        ind)

                pixel_spacing = \
                    get_reoriented_PS(
                        image_data.frames[frame],
                        orientation)
                rescaled = True
        else:
            # Case 2
            dataset = None
            image = reorient_from_axial(
                image_data[frame],
                orientation,
                ind)
            pixel_spacing = None
            rescaled = False

    elif is_series or is_3d_array:
        # Cases 3 & 4: Image series

        if ind is None:
            raise ValueError(
                "ind must be provided for image series")

        if is_series:
            # Case 3
            if not reoriented:
                # For performance, no need to generate stack,
                # but pixel values aren't rescaled
                dataset = image_data.slices[ind]
                image = dataset.pixel_array
                pixel_spacing = get_PS(dataset)
                rescaled = False
            else:
                # Used for windowing
                dataset = image_data.slices[0]

                image = reorient_from_axial(
                    image_data.goc_stack(), orientation, ind)
                pixel_spacing = get_reoriented_PS(
                    image_data.slices, orientation)
                rescaled = True
        else:
            # Case 4
            dataset = None
            image = reorient_from_axial(
                image_data,
                orientation,
                ind)
            pixel_spacing = None
            rescaled = False

    elif is_dataset or is_2d_array:
        # Cases 5 & 6: Images

        if is_dataset:
            # Case 5
            dataset = image_data
            image = dataset.pixel_array
            pixel_spacing = get_PS(dataset)
            rescaled = False
        else:
            # Case 6
            dataset = None
            image = image_data
            pixel_spacing = None
            rescaled = False
    else:
        raise ValueError("Invalid input data")

    return dataset, image, pixel_spacing, rescaled


def _round_to_int(x):
    """
    Point coordinates: In pixels from top-left corner of complete
    image (before FOV)
    """

    return np.round(x).astype(int)
