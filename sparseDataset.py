import six
from scipy import sparse
import numpy as np

class SparseDataset(object):

    """Dataset of a tuple of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.

    Args:
        datasets: Underlying datasets. The ``i``-th one is used for the
            ``i``-th item of each example. All datasets must have the same
            length.

    """

    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = datasets[0].shape[0]
        for i, dataset in enumerate(datasets):
            if dataset.shape[0] != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        # convert to dense matrix for use
        batches = [np.asarray(dataset[index].todense()) if sparse.issparse(dataset)
                   else dataset[index] for dataset in self._datasets]
        # batches = [dataset[index] for dataset in self._datasets] # old
        if isinstance(index, slice):
            length = batches[0].shape[0]
            return [tuple([batch[i] for batch in batches])
                    for i in six.moves.range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length

