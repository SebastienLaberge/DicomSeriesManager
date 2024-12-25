"""
Module defining classes for storing DICOM images
"""

from collections.abc import Sequence
from itertools import groupby
from numbers import Number
from typing import Union

from numpy import array, float32, stack
from pydicom import Dataset

from .reader import DicomDirContent

# Target type for pixels/voxels
pixel_type = float32


class BaseSeries:
    """
    Base class for series of DICOM images that can be stacked into
    an ndarray
    """

    def __init__(self):

        self._stack = None

    def goc_stack(self):
        """
        Get or create stack
        """
        if self._stack is None:

            self._stack = self._create_stack()

        return self._stack

    def is_multivolume(self) -> bool:
        """
        Return whether this is a multi-volume
        """
        return NotImplemented

    def get_vol_shape(self, frame: int = 0) -> array:
        """
        Return shape of the 3D volume for a given frame
        """
        return NotImplemented

    def get_number_of_frames(self) -> int:
        """
        Return he number of frames
        """
        return NotImplemented

    def get_number_of_slices(self, frame: int = 0) -> int:
        """
        Return number of slices for a given frame
        """
        return NotImplemented

    def get_frame(self, frame: int = 0) -> Sequence[Dataset]:
        """
        Return a series of datasets corresponding to a given frame
        """
        return NotImplemented

    def get_dataset(self, ind: int, frame: int = 0) -> Dataset:
        """
        Return one of the datasets
        """
        return NotImplemented

    def _create_stack(self) -> array:
        """
        Return ndarray representing a stack of stored elements
        """
        return NotImplemented


class Series(BaseSeries):
    """
    Class for series of images that can be stacked into a 3D array
    """

    def __init__(self, slices: Sequence[Dataset]):
        """
        Initialize from sequence of slices
        """

        super().__init__()

        self.slices = slices

    def is_multivolume(self) -> bool:

        return False

    def get_vol_shape(self, _: int = 0) -> array:

        # Slice dimensions assumed to be the same size as all
        # the others
        exemplar = self.get_dataset(0)

        depth = self.get_number_of_slices()
        height = exemplar.Columns
        width = exemplar.Rows

        return array([depth, height, width])

    def get_number_of_frames(self) -> int:

        return 1

    def get_number_of_slices(self, _: int = 0) -> int:

        return len(self.slices)

    def get_frame(self, _: int = 0) -> Sequence[Dataset]:

        return self.slices

    def get_dataset(self, ind: int, _: int = 0) -> Dataset:

        return self.slices[ind]

    def _create_stack(self):
        """
        Stack slices into 3D array: [slice, row, column]
        """

        mapping = _get_stacker(_get_pixel_data)

        return mapping(self.slices)


class DynSeries(BaseSeries):
    """
    Class for time series of 3D frames that can be stacked into
    a 4D array
    """

    # Initialize from sequence of time frames along with their
    # timestamps
    def __init__(self,
                 frames: Sequence[Sequence[Dataset]],
                 timestamps: Sequence[Number]):

        super().__init__()

        assert (len(frames) == len(timestamps))

        # Convert sequence of timestamps to ndarray
        timestamps = array(timestamps, dtype=pixel_type)

        # Convert timestamp values from milliseconds to seconds
        timestamps /= 1000.0

        self.frames = frames
        self.timestamps = timestamps

    def is_multivolume(self) -> bool:

        return True

    def get_vol_shape(self, frame: int = 0):

        # First slice assumed to be the same size as all the
        # others
        exemplar = self.get_dataset(0, frame)

        depth = self.get_number_of_slices(frame)
        height = exemplar.Columns
        width = exemplar.Rows

        return array([depth, height, width])

    def get_number_of_frames(self) -> int:

        return len(self.frames)

    def get_number_of_slices(self, frame: int = 0) -> int:

        return len(self.frames[frame])

    def get_frame(self, frame: int = 0) -> Sequence[Dataset]:

        return self.frames[frame]

    def get_dataset(self, ind: int, frame: int = 0) -> Dataset:

        return self.frames[frame][ind]

    def _create_stack(self):
        """
        Stack frames into 4D array: [frame, slice, row, column]
        Warning: Only works if volumes have the same size in
        each frame
        """

        mapping = _get_stacker(_get_stacker(_get_pixel_data))

        return mapping(self.frames)

    @staticmethod
    def is_dyn_series(slices):
        """
        Determine if a series of slices form a dynamic series
        """

        timeAttr = 'FrameReferenceTime'
        def hasTimeAttr(slice): return hasattr(slice, timeAttr)

        return all(map(hasTimeAttr, slices))

    @staticmethod
    def regroup_slices(slices):
        """
        Regroup slices from content file according to timestamp
        Note: It is assumed that slices are already sorted
        according to timestamp
        """

        timeAttr = 'FrameReferenceTime'

        def getTimeAttr(slice):
            return getattr(slice, timeAttr, None)

        def to_float(x):
            return float(x) if x is not None else None

        groups = groupby(slices, key=getTimeAttr)
        data = [(list(group), to_float(timestamp))
                for timestamp, group in groups]

        return tuple(zip(*data))


def series_factory(
        content: DicomDirContent,
        series_specifier: Union[int, str],
        get_dyn_series: bool = True) -> BaseSeries:
    """
    From the content of a DICOM directory, create an object
    representing a single series specified by its index or by
    its series description string and if requested, return any
    series that happens to be dynamic using a dedicated type
    """

    if isinstance(series_specifier, int):
        slices = content.read_series_files(series_specifier)

    elif isinstance(series_specifier, str):
        slices = content.read_series_files_from_description(
            series_specifier)
    else:
        raise TypeError("Parameter series_specifier must be of "
                        "type str or int")

    return DynSeries(*DynSeries.regroup_slices(slices)) \
        if get_dyn_series and DynSeries.is_dyn_series(slices) \
        else Series(slices)


def _get_pixel_data(dataset: Dataset):
    """
    Get pixel data from dataset as numpy array with the
    following pretreatment:
    1) Pixel values cast to the target type
    2) Pixel values rescaled using parameters from the DICOM
       header
    """

    # Get pixel array and cast its elements to target type
    pixel_array = dataset.pixel_array.astype(pixel_type)

    # Get rescale parameters and cast them to target type
    slope = pixel_type(dataset.RescaleSlope)
    intercept = pixel_type(dataset.RescaleIntercept)

    # Apply rescaling to pixel array
    return slope*pixel_array + intercept


def _get_stacker(get_array):
    """
    Get a lambda that applies a function get_array returning an
    ndarray to the elements of an input sequence and stacks the
    resulting arrays
    """

    return lambda seq: stack(
        [get_array(elem) for elem in seq], axis=0)
