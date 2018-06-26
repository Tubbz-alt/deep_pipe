import numpy as np

from .segmentation import SegmentationFromCSV


# We need this class because in the original data segm values are [0, 1, 2, 4]
class Brats2017(SegmentationFromCSV):
    def __init__(self, data_path, metadata_rpath='metadata.csv'):
        super().__init__(data_path=data_path, modalities=['t1', 't1ce', 't2', 'flair'], target='segm',
                         metadata_rpath=metadata_rpath)

    def load_segm(self, identifier):
        segm = super().load_segm(identifier)
        segm[segm == 4] = 3
        return np.uint8(segm)


segm_decoding_matrix = np.array([[0, 0, 0],
                                 [1, 1, 0],
                                 [1, 0, 0],
                                 [1, 1, 1]
                                 ], dtype=bool)

# For Brats 2015
# segm2msegm = np.array([
#     [0, 0, 0],
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 1, 1]
# ], dtype=bool)
