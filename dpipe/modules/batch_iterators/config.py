from .patch_3d_stratified import make_3d_patch_stratified_iter
from .slices import *

batch_iter_name2batch_iter = {
    '3d_patch_strat': make_3d_patch_stratified_iter,
    'slices': make_slices_iter,
    'multiple_slices': make_multiple_slices_iter,
}
