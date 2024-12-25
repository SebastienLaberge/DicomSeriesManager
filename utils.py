"""
Module containing functions to compute useful quantities from
DICOM datasets

Abbreviations:
  IPP: ImagePositionPatient
  IOP: ImageOrientationPatient
  PS: PixelSpacing
"""

from typing import Optional, Sequence

import numpy as np

from pydicom.dataset import Dataset
from pydicom.multival import MultiValue


def get_IPP(dataset: Dataset):
    """
    Get position of center of first pixel in space as [x, y, z]
    """

    return np.array(
        dataset.ImagePositionPatient,
        dtype=np.float32)


def get_IOP(dataset: Dataset):
    """
    Get image orientation in space as two unit vectors:
    ([horizontal_x, horizontal_y, horizontal_z],
     [vertical_x, vertical_y, vertical_z])
    """

    return (np.array(dataset.ImageOrientationPatient[:3],
                     dtype=np.float32),
            np.array(dataset.ImageOrientationPatient[3:],
                     dtype=np.float32))


def get_normal(dataset: Dataset):
    """
    Get unit vector normal to image plane and pointing from
    observer to screen
    """

    IOP = get_IOP(dataset)

    return np.cross(IOP[0], IOP[1])


def get_PS(dataset: Dataset):
    """
    Get pixel spacing as [vertical, horizontal]
    """

    return np.array(dataset.PixelSpacing, dtype=np.float32)


def get_slice_spacing(
        dataset_slice_0: Dataset,
        dataset_slice_1: Dataset):
    """
    Get distance between two slices
    """

    return np.linalg.norm(
        get_IPP(dataset_slice_1) -
        get_IPP(dataset_slice_0))


def check_slice_spacing_consistency(
        dataset_list: Sequence[Dataset]):
    """
    For a stack of slices provided as a list of datasets, check
    that the distance between neighboring slices is the same
    everywhere
    """

    spacings = [get_slice_spacing(slice0, slice1)
                for slice0, slice1 in
                zip(dataset_list[:-1], dataset_list[1:])]

    max_diff = np.diff(spacings).max()

    return max_diff <= np.finfo(max_diff.dtype).eps


def get_point_to_position_matrix(dataset: Dataset):
    """
    From a dataset representing an image, get the matrix M that
    can be used to get the R = [x,y,z,1]^T world coordinates of
    the pixel given by intrinsic coordinates I = [j,i,0,1]^T.
    (R = MI)
    """

    IPP = get_IPP(dataset)
    IOP = get_IOP(dataset)
    PS = get_PS(dataset)

    matrix = np.eye(4)

    matrix[0:3, 0] = IOP[0] * PS[1]
    matrix[0:3, 1] = IOP[1] * PS[0]
    matrix[0:3, 3] = IPP

    return matrix


def get_point_to_position_matrix_with_depth(
        dataset_slice_0: Dataset,
        dataset_slice_1: Dataset):
    """
    From two datasets representing the first two slices of a
    volume, get the matrix M that can be used to get the
    R = [x,y,z,1]^T world coordinates of the pixel given by
    intrinsic coordinates I = [j,i,slice,1]^T.
    (R = MI)
    """

    point_to_position_matrix = \
        get_point_to_position_matrix(dataset_slice_0)

    # Modify third column to allow displacement along depth
    # dimension
    normal = get_normal(dataset_slice_0)
    slice_spacing = get_slice_spacing(
        dataset_slice_0,
        dataset_slice_1)
    point_to_position_matrix[2, 0:3] = normal * slice_spacing

    return point_to_position_matrix


def get_window_limits(
        window,
        dataset: Optional[Dataset],
        image: Optional[np.ndarray],
        rescaled: bool):
    """
    Given a specifier for a display window, get the low and high
    pixel values of that window

    Case A (window is None and an image is provided):
        Return the minimum and maximum values of the input image.

    Case B (window is a list of two values)
        Interpret the window as (center, width).

    Case C (window is an index and a dataset is provided)
        Take the window from the dataset specified by the index
    """

    if window is None:

        # Case A
        assert image is not None

        v_min = image.min()
        v_max = image.max()

        return v_min, v_max

    # Get window center and window width
    if hasattr(window, '__getitem__'):

        # Case B
        window_center = window[0]
        window_width = window[1]

    else:
        # Case C
        assert dataset is not None

        def get_window_attr(dataset, window, keyword):

            attr = getattr(dataset, keyword)

            if isinstance(attr, MultiValue):

                if window < 0 or window >= len(attr):

                    raise ValueError(
                        "window index out of range")

                attr = attr[window]

            return np.float32(attr)

        window_center = \
            get_window_attr(dataset, window, "WindowCenter")

        window_width = \
            get_window_attr(dataset, window, "WindowWidth")

    # Compute minimum and maximum pixel values
    v_min = window_center - window_width / 2
    v_max = window_center + window_width / 2

    # Reverse rescaling on vmin, vmax if image pixel values
    # aren't rescaled
    # TODO: CHECK THIS IS DONE RIGHT FOR ARRAY INPUTS
    if not rescaled and dataset is not None:

        slope = np.float64(dataset.RescaleSlope)
        intercept = np.float64(dataset.RescaleIntercept)

        v_min = (v_min - intercept) / slope
        v_max = (v_max - intercept) / slope

    return v_min, v_max


_SpatialCoord = np.float64


def get_slice_limits(FOV, im_shape, pixel_spacing=None):
    """
    Convert an FOV to a range of pixels along each dimension.
    Two ranges are returned, one for x and one for y.
    Each range contains a minimum and a maximum index value.

    If pixel_spacing is provided, the components of FOV are
    assumed to be expressed in real-world coordinates (mm)
    instead of pixel indices.
    """

    if len(FOV) != 4:

        raise ValueError('FOV must have 4 elements')

    width = FOV[0]
    height = FOV[1]
    offset_x = FOV[2]
    offset_y = FOV[3]

    width_max = im_shape[1]
    height_max = im_shape[0]

    # If pixel spacing is provided, convert FOV dimensions
    # to pixel indices. At the same time, treat None values for
    # width and height as representing the maximum possible
    # width and height respectively.
    if pixel_spacing is not None:

        width = width_max \
            if width is None \
            else width / pixel_spacing[1]
        height = height_max \
            if height is None \
            else height / pixel_spacing[0]

        offset_x = offset_x / pixel_spacing[1]
        offset_y = offset_y / pixel_spacing[0]
    else:
        if width is None:
            width = width_max

        if height is None:
            height = height_max

    # Get the FOV dimensions and offset as integer values
    FOV_dims = _SpatialCoord([width, height]).round()
    FOV_offset = _SpatialCoord([offset_x, offset_y]).round()

    # Get the coordinates of the center pixel of the FOV
    im_center = _SpatialCoord([width_max // 2, height_max // 2])
    FOV_center = im_center + FOV_offset

    # Get inclusive range of pixel coordinates defining FOV
    # along each axis
    x_range = FOV_center[0] + \
        _SpatialCoord([
            -(FOV_dims[0] // 2),
            (FOV_dims[0] - 1) // 2])
    y_range = FOV_center[1] + \
        _SpatialCoord([
            -(FOV_dims[1] // 2),
            (FOV_dims[1] - 1) // 2])

    # Clamp coordinates to image boundaries
    def clamp_to_boundary_x(x_coord):
        return int(
            np.min([
                np.max([x_coord, 0]),
                width_max - 1]))

    def clamp_to_boundary_y(y_coord):
        return int(
            np.min([
                np.max([y_coord, 0]),
                height_max - 1]))

    x_range = list(map(clamp_to_boundary_x, x_range))
    y_range = list(map(clamp_to_boundary_y, y_range))

    return x_range, y_range
