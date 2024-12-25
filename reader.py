"""
Module for reading DICOM files and generating DICOM series
objects
"""

from os import walk
from os.path import join
from pickle import dump, load
from pprint import pformat
from typing import List

from numpy import array, dot, cross

from pydicom import dcmread, FileDataset
from pydicom.misc import is_dicom


class DicomDirContent:
    """
    Class describing the content of a directory containing DICOM
    files and allowing easy access to the paths of the files
    constituting any given series.
    """

    def __init__(self, dicom_dir_path: str):
        """
        Reading a directory containing DICOM files
        """

        # Read info from all DICOM series in directory
        series_dict = _SeriesDict(dicom_dir_path)

        # Get series data to be saved in content file
        self.series_list = \
            [_SeriesFiles(series_info)
             for series_info in series_dict.values()]

    def read_series_files(self, series_index: int):
        """
        For a given series specified by its index, read data
        for that series as a list of PyDicom datasets
        """

        assert series_index < len(self.series_list)

        series_files = self.series_list[series_index].files

        return [dcmread(file) for file in series_files]

    def read_series_files_from_description(
            self,
            series_description: str):
        """
        For a given series specified by its description string,
        read data for that series as a list of PyDicom datasets

        Note: This more reliable than read_series_files for
        finding a given series in a scripted manner since the
        order in which series are read may vary between systems.
        """

        found_series_index = None
        for series_index, series in enumerate(self.series_list):
            if series.info['SeriesDescription'] \
               == series_description:
                found_series_index = series_index
                break

        if found_series_index is None:
            raise ValueError(
                f"No series has the following value of the tag "
                f"SeriesDescription: {series_description}")

        return self.read_series_files(found_series_index)

    def save(self, content_file_path: str):
        """
        Save the object for quick retrieval
        """

        with open(content_file_path, 'wb') as fp:
            dump(self, fp)

    @staticmethod
    def load(content_file_path: str):
        """
        Load a previously saved object
        """

        with open(content_file_path, 'rb') as fp:
            file = load(fp)

        return file


class _FileInfo:
    """
    Class for extracting and storing relevant metadata from a
    DICOM file
    """

    def __init__(self, file_path: str):

        # Read dataset with metadata only
        dataset = dcmread(file_path, stop_before_pixels=True)

        # Value to indicate that the file is discarded
        # Warning: For discarded files, other attributes are not
        # initialized
        self.series_UID = None

        # Discard DICOMDIR
        if not isinstance(dataset, FileDataset):
            return

        spatial_info = self.SpatialInfo(dataset)

        # Discard files that are not from a volume
        if not spatial_info.is_volume:
            return

        self.series_UID = dataset.SeriesInstanceUID
        self.file_path = file_path
        self.spatial = spatial_info
        self.frame = self.FrameInfo(dataset)
        self.general = self.GeneralInfo(dataset)

    class Info(dict):
        """
        Base class for dictionaries containing specific DICOM
        tag values
        """

        def extract_tags(self, dataset, tag_list):
            """
            Fill dictionary with values of specified tags from
            dataset
            """

            # Note: Tags are set to None if they are not found
            # in the dataset
            tags = \
                {tag: getattr(dataset, tag, None)
                 for tag in tag_list}
            self.update(tags)

    class SpatialInfo(Info):
        """
        Information on the spatial orientation of an image
        """

        spatial_tags = ['ImagePositionPatient',
                        'ImageOrientationPatient',
                        'PixelSpacing']

        def __init__(self, dataset):

            self.extract_tags(dataset, self.spatial_tags)

            # Check if file contains a slice from a volume
            self.is_volume = \
                all((v is not None for v in self.values()))

    class FrameInfo(Info):
        """
        Information on the frame an image belongs to
        """

        frame_tags = ['FrameReferenceTime']

        # Note:
        # Dynamic volumes often also have tags NumberOfSlices
        # and NumberOfTimeSlices but non-dynamic volumes may be
        # multi-frame too and not have those tags.

        def __init__(self, dataset):

            self.extract_tags(dataset, self.frame_tags)

            # Check if file contains a slice from a multivolume
            self.is_multivolume = \
                all((v is not None for v in self.values()))

    class GeneralInfo(Info):
        """
        Information on the context of a series acquisition
        """

        general_tags = ['Modality',
                        'SeriesDescription',
                        'SeriesType',
                        'BodyPartExamined',
                        'ProtocolName']

        def __init__(self, dataset):

            self.extract_tags(dataset, self.general_tags)


class _SeriesDict(dict):
    """
    Dictionary mapping a series UID to a list of _FileInfo
    objects
    """

    def __init__(self, dicom_dir_path: str):

        # Get paths to all DICOM files in input directory
        dicom_files = self.get_dicom_file_paths(dicom_dir_path)

        # Store info of all files that contain a volume slice
        for file_path in dicom_files:

            info = _FileInfo(file_path)

            series_UID = info.series_UID
            if series_UID is None:
                continue

            if series_UID not in self:
                self[series_UID] = [info]
            else:
                self[series_UID].append(info)

        # Put slices in the right order for each series
        for series_info in self.values():

            self.sort_series(series_info)

    @staticmethod
    def get_dicom_file_paths(dicom_dir_path: str) -> List[str]:
        """
        Return list of full paths to all DICOM files in input
        directory
        """

        return [file_path
                for root, _, files in walk(dicom_dir_path)
                for file_path in [join(root, name)
                                  for name in files]
                if is_dicom(file_path)]

    @staticmethod
    def sort_series(series_info: List[_FileInfo]):
        """
        Sort files of a series into (multi)volumes
        """

        # Info for first slice
        info_slice0 = series_info[0]

        # Lambda to get IOP tag for a given slice
        def get_IOP(info):
            return info.spatial['ImageOrientationPatient']

        # Get image orientation for the first slice
        IOP_slice0 = get_IOP(info_slice0)

        # Assert that image orientation is the same for every
        # slice
        assert (all(get_IOP(info) for info in series_info[1:]))

        # Compute normal vector
        N = cross(array(IOP_slice0[:3]), array(IOP_slice0[3:]))

        # Lambda to get IPP tag for a given slice
        def get_IPP(info):
            return info.spatial['ImagePositionPatient']

        # Get image position for the first slice
        IPP_slice0 = array(get_IPP(info_slice0))

        # Lambda to compute slice location
        def get_slice_location(info):
            return dot(get_IPP(info) - IPP_slice0, N)

        # Get sorting key
        if info_slice0.frame.is_multivolume:
            # Lambda to get frame timestamp
            def get_timestamp(info):
                return float(info.frame['FrameReferenceTime'])

            def k(info):
                return (get_timestamp(info),
                        get_slice_location(info))
        else:
            k = get_slice_location

        # Sort slices according to timestamp and slice location
        series_info.sort(key=k)


class _SeriesFiles:
    """
    Class for storing paths to files of a series along with
    some info
    """

    def __init__(self, series_info: List[_FileInfo]):

        # Take general info from first slice to represent the
        # whole series
        self.info = dict(series_info[0].general)

        # Get list of all files for slices in the series
        self.files = [info.file_path for info in series_info]

    def __repr__(self):

        info = {k: v
                for k, v in self.info.items()
                if v is not None}
        info['n_slices'] = len(self.files)

        return f'\n{pformat(info)}\n'

    def __len__(self):

        return len(self.files)
