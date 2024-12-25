"""
Module for obtaining slices and pixel properties according to
various orientations
"""

import numpy as np

from .utils import get_PS, get_slice_spacing


def reorient_from_axial(vol, orientation, ind):
    """
    Get a slice of an axial volume (3D ndarray) along a given
    orientation
    Note: Slice indices are increasing while the slice is moving
    away from the observer
    """

    switcher = {
        'Axial': lambda ind: vol[ind],
        'Coronal': lambda ind: np.flip(vol[:, ind], 0),
        'Sagittal': lambda ind: np.flip(vol[:, :, -ind-1], 0)
    }

    f = switcher.get(orientation.capitalize())

    if f is None:
        raise ValueError(f'Invalid orientation "{orientation}"')

    return f(ind)


def get_reoriented_slice_index(
        indices_3D,
        vol_shape,
        orientation):
    """
    From the indices of a 3D voxel, get a slice index for a
    given orientation
    """

    signed_perm = _get_signed_perm(orientation)

    # Depth dimension
    d = signed_perm[2]

    slice_index = _get_processed_index(indices_3D, vol_shape, d)

    return slice_index


def get_reoriented_marker_indices(
        indices_3D,
        vol_shape,
        orientation):
    """
    Get pixel spacing ([interline, intercolumn]) for a
    reoriented slice of a dataset list
    From a 3D voxel index, get a marker index on the image plane
    (i, j) for a given orientation
    """

    signed_perm = _get_signed_perm(orientation)

    h = signed_perm[0]  # Horizontal dimension
    v = signed_perm[1]  # Vertical dimension

    i = _get_processed_index(indices_3D, vol_shape, v)
    j = _get_processed_index(indices_3D, vol_shape, h)

    return [i, j]


def get_reoriented_n_slices(vol_shape, orientation):
    """
    For a given orientation, get the number of slices along
    that dimension
    """

    signed_perm = _get_signed_perm(orientation)

    d = signed_perm[2]  # Depth dimension

    return _get_processed_size(vol_shape, d)


def get_reoriented_im_shape(vol_shape, orientation):
    """
    For a given orientation, get the shape of the image
    """

    signed_perm = _get_signed_perm(orientation)

    h = signed_perm[0]  # Horizontal dimension
    v = signed_perm[1]  # Vertical dimension

    h_size = _get_processed_size(vol_shape, h)
    v_size = _get_processed_size(vol_shape, v)

    return v_size, h_size


def get_reoriented_FOV(FOV_size_3D, FOV_offset_3D, orientation):
    """
    From a 3D FOV, get a FOV compatible with the show function
    for a given orientation
    Note: Input triplets contain components defined according
    to the patient's referential
    """

    signed_perm = _get_signed_perm(orientation)

    h = signed_perm[0]  # Horizontal dimension
    v = signed_perm[1]  # Vertical dimension

    h0 = np.abs(h) - 1
    v0 = np.abs(v) - 1

    hs = np.sign(h)
    vs = np.sign(v)

    return [FOV_size_3D[h0],
            FOV_size_3D[v0],
            FOV_offset_3D[h0]*hs,
            FOV_offset_3D[v0]*vs]


def get_reoriented_PS(dataset_list, orientation):

    original_PS = get_PS(dataset_list[0])

    switcher = {
        'Axial': lambda: original_PS,

        'Coronal': lambda: np.array([
            get_slice_spacing(dataset_list[0], dataset_list[1]),
            original_PS[1]], dtype=np.float32),

        'Sagittal': lambda: np.array([
            get_slice_spacing(dataset_list[0], dataset_list[1]),
            original_PS[0]], dtype=np.float32)
    }

    f = switcher.get(orientation.capitalize())

    if f is None:
        raise ValueError(f'Invalid orientation "{orientation}"')

    return f()


# Orientations of fixed spatial axes defining the patient's
# referential according to the DICOM convention:
# Axis 1 (x): Right->Left
# Axis 2 (y): Anterior->Posterior
# Axis 3 (z): Inferior->Superior
#
# Image axis associated with each position in a signed
# permutation expressed as a triplet:
# triplet[0]: Increasing column index (Left->Right)
# triplet[1]: Increasing line index (Top->Bottom)
# triplet[2]: Increasing slice index (Away from the observer)
#
# Negative sign means an inverted spatial axis


def _get_signed_perm(orientation):
    """
    Get signed permutations representing orientation of image
    with respect to patient
    """

    _signed_perm_dict = {
        'Axial': (1, 2, 3),
        'Coronal': (1, -3, 2),
        'Sagittal': (2, -3, -1)
    }

    signed_perm = _signed_perm_dict.get(orientation.capitalize())

    if signed_perm is None:
        raise ValueError(f'Invalid orientation "{orientation}"')

    return signed_perm


def _get_processed_index(
        indices_3D,
        vol_shape,
        signed_perm_elem):
    """
    Select an index and conditionally reverse it according to
    a signed permutation element
    Note: Indices are defined and ordered according to the
    patient's referential: [x, y, z]
    Warning: Elements in volume shape are in reverse order:
             [z, y, x]
    Note: Volume shape is only used if the selected index is
    reversed
    """

    dim_index = np.abs(signed_perm_elem) - 1

    dim_inversion = signed_perm_elem < 0

    index = indices_3D[dim_index]

    # Revert index if necessary
    if dim_inversion:
        # Get number of index values along selected dimension,
        # taking into account the reverse order of elements in
        # the volume shape
        dim_reverted = 2 - dim_index
        n = vol_shape[dim_reverted]

        index = n - 1 - index

    return index


def _get_processed_size(vol_shape, signed_perm_elem):

    dim_index = np.abs(signed_perm_elem) - 1

    # Take into account the reverse order of elements in the
    # volume shape
    dim_reverted = 2 - dim_index

    return vol_shape[dim_reverted]
