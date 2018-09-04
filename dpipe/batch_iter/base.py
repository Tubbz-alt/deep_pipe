from abc import abstractmethod, ABC
from itertools import islice
from contextlib import contextmanager
from typing import Iterable, Callable, Union

import pdp

__all__ = ['BatchIter', 'make_batch_iter_from_finite', 'make_batch_iter_from_infinite', 'make_infinite_batch_iter']


@contextmanager
def build_contextmanager(o):
    yield o


def maybe_build_contextmanager(o):
    """If input has no context manager on it's own, turn it into object with empty context manager."""
    if hasattr(o, '__enter__'):
        return o
    else:
        return build_contextmanager(o)


class BatchIter(ABC):
    """Interface for training functions, that unifies interface for infinite and finite batch generators.

    Examples
    --------
    >>> # BatchIter should be created from one of the implementations
    >>> batch_iter : BatchIter = None
    >>> with batch_iter:
    >>>     for epoch in range(10):
    >>>         for x, y in batch_iter:
    >>>             pass

    """

    @abstractmethod
    def __iter__(self):
        pass

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BatchIterRepeater(BatchIter):
    def __init__(self, get_batch_iter):
        self.get_batch_iter = get_batch_iter
        self.batch_iter = None

    def __iter__(self):
        assert self.batch_iter is None, 'Iterator has already been open'
        self.batch_iter = maybe_build_contextmanager(self.get_batch_iter())
        with self.batch_iter:
            yield from self.batch_iter
        self.batch_iter = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # We need this part in case during batch iteration there was an error in the main thread
        # This part could be dangerous. If there was an error in the main thread, self.batch_iter.__exit__ will
        # be called twice. pdp backend just ignores second exit, so, no problem here.
        if self.batch_iter is not None:
            result = self.batch_iter.__exit__(exc_type, exc_val, exc_tb)
            self.batch_iter = None
            return result
        else:
            return False


class BatchIterSlicer(BatchIter):
    def __init__(self, get_batch_iter, n_iters_per_epoch):
        self.infinite_batch_iter = maybe_build_contextmanager(get_batch_iter())
        self.n_iters_per_epoch = n_iters_per_epoch

    def __iter__(self):
        yield from islice(self.infinite_batch_iter, self.n_iters_per_epoch)

    def __enter__(self):
        self.infinite_batch_iter.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.infinite_batch_iter.__exit__(exc_type, exc_val, exc_tb)


def make_batch_iter_from_finite(get_batch_iter):
    return BatchIterRepeater(get_batch_iter)


def make_batch_iter_from_infinite(get_batch_iter, n_iters_per_epoch):
    return BatchIterSlicer(get_batch_iter, n_iters_per_epoch)


def make_infinite_batch_iter(source: Union[Iterable, pdp.Source], *transformers: Union[Callable, pdp.One2One],
                             batch_size: int, n_iters_per_epoch: int, buffer_size: int = 3):
    """
    Combine `source` and `transformers` into a batch iterator that yields batches of size `batch_size`.

    Parameters
    ----------
    source: Iterable, pdp.Source
        an infinite iterable.
    transformers: Callable, pdp.One2One
        a callable that transforms the objects generated by the previous element of the pipeline.
    batch_size: int
    n_iters_per_epoch: int
        how many batches to yield before exhaustion.
    buffer_size: int, optional
    """

    def wrap(o):
        if not isinstance(o, pdp.interface.TransformerDescription):
            o = pdp.One2One(o, buffer_size=buffer_size)
        return o

    def combine_batches(inputs):
        return tuple(zip(*inputs))

    if not isinstance(source, pdp.Source):
        source = pdp.Source(source, buffer_size=buffer_size)

    pipeline = pdp.Pipeline(source, *map(wrap, transformers), pdp.Many2One(chunk_size=batch_size, buffer_size=3),
                            pdp.One2One(combine_batches, buffer_size=buffer_size))

    return BatchIterSlicer(lambda: pipeline, n_iters_per_epoch)


wrap_infinite_pipeline = make_infinite_batch_iter