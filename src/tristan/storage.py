"""
Tools for storage.

Provides IAddArray, a subclass of zarr.Array with a method for thread- or process-safe
in-place addition, using Zarr's existing synchronisation.  This is useful for
concurrently adding partial contributions to an array where contention cannot be
avoided.  Synchronisation is performed by chunk-wise locking, to permit uncontended
chunk addition to continue unimpeded.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import zarr
from numcodecs.compat import ensure_ndarray_like
from zarr.errors import ReadOnlyError
from zarr.indexing import (
    CoordinateIndexer,
    check_fields,
    check_no_multi_fields,
    is_scalar,
)
from zarr.util import NoLock, check_array_shape

nolock = NoLock()


def create_cache(output_file: Path | str, shape: tuple[int, ...]) -> zarr.Array:
    """
    Make a Zarr array of zeros, suitable for using as an image binning cache.

    The array has the path "data" and has shape ``shape`` and is chunked by image, i.e.
    the chunk shape will be ``(*(1,) * len(shape[:-2]), *shape[:-2])``.

    The underlying store is a zarr.DirectoryStore, which is re-initialised with zero
    values if there exists a store with the same name.  The chunks have thread-safe
    locking.

    Args:
        output_file:  Output file name.  Any file extension will be replaced with .zarr.
        shape:        The shape of the cache array.

    Returns:
        The Zarr array.
    """
    # Store in a zarr.TempStore, which will be torn down at exit.
    output_file = Path(output_file)
    unique = uuid.uuid4()
    prefix = f"{output_file.stem}-{unique}"
    store = zarr.TempStore(prefix=prefix, suffix=".zarr", dir=".")

    chunks = *(1,) * len(shape[:-2]), *shape[-2:]
    array = zarr.zeros(
        store=store,
        path="data",
        shape=shape,
        chunks=chunks,
        dtype=np.int32,
        overwrite=True,
    )
    return IAddArray(
        store=array.store, path=array.path, synchronizer=zarr.ThreadSynchronizer()
    )


class IAddArray(zarr.Array):
    """A Zarr array with thread- or process-safe in-place addition."""

    def iadd_coordinate_selection(self, selection, value, fields=None):
        """
        Add to a selection of individual items, by providing the indices (coordinates)
        for each item to be modified.

        Borrows heavily from zarr.Array.set_coordinate_selection.

        Parameters
        ----------
        selection : tuple
            An integer (coordinate) array for each dimension of the array.
        value : scalar or array-like
            Value to be added into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            set data for.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> from tristan.storage import IAddArray
            >>> import zarr
            >>> z = zarr.zeros((5, 5), dtype=int)
            >>> a = IAddArray(z.store, z.path)

        Add to a selection of items::

            >>> a.iadd_coordinate_selection(([1, 4], [1, 4]), 1)
            >>> a[...]
            array([[0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])
            >>> a.iadd_coordinate_selection(([1, 2], [1, 3]), [3, 2])
            >>> a[...]
            array([[0, 0, 0, 0, 0],
                   [0, 4, 0, 0, 0],
                   [0, 0, 0, 2, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])

        Notes
        -----
        Coordinate indexing is also known as point selection, and is a form of
        vectorized or inner indexing.

        Slices are not supported. Coordinate arrays must be provided for all dimensions
        of the array.

        """

        # guard conditions
        if self._read_only:
            raise ReadOnlyError

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # setup indexer
        indexer = CoordinateIndexer(selection, self)

        # handle value - need ndarray-like flatten value
        if not is_scalar(value, self._dtype):
            try:
                value = ensure_ndarray_like(value)
            except TypeError:
                # Handle types like `list` or `tuple`
                value = np.array(value, like=self._meta_array)
        if hasattr(value, "shape") and len(value.shape) > 1:
            value = value.reshape(-1)

        self._add_selection(indexer, value, fields=fields)

    def _add_selection(self, indexer, value, fields=None):
        """
        Essentially a copy of self._set_selection.

        The only change is that _chunk_setitem is replaced with _chunk_add.
        """
        # We iterate over all chunks which overlap the selection and thus contain data
        # that needs to be replaced. Each chunk is processed in turn, extracting the
        # necessary data from the value array and storing into the chunk array.

        # N.B., it is an important optimisation that we only visit chunks which overlap
        # the selection. This minimises the number of iterations in the main for loop.

        # check fields are sensible
        check_fields(fields, self._dtype)
        fields = check_no_multi_fields(fields)

        # determine indices of chunks overlapping the selection
        sel_shape = indexer.shape

        # check value shape
        if sel_shape == ():
            # setting a single item
            pass
        elif is_scalar(value, self._dtype):
            # setting a scalar value
            pass
        else:
            if not hasattr(value, "shape"):
                value = np.asanyarray(value, like=self._meta_array)
            check_array_shape("value", value, sel_shape)

        # iterate over chunks in range
        if (
            not hasattr(self.chunk_store, "setitems")
            or self._synchronizer is not None
            or any(map(lambda x: x == 0, self.shape))
        ):
            # iterative approach
            for chunk_coords, chunk_selection, out_selection in indexer:
                # extract data to store
                if sel_shape == ():
                    chunk_value = value
                elif is_scalar(value, self._dtype):
                    chunk_value = value
                else:
                    chunk_value = value[out_selection]
                    # handle missing singleton dimensions
                    if indexer.drop_axes:
                        item = [slice(None)] * self.ndim
                        for a in indexer.drop_axes:
                            item[a] = np.newaxis
                        item = tuple(item)
                        chunk_value = chunk_value[item]

                # put data
                self._chunk_add(
                    chunk_coords, chunk_selection, chunk_value, fields=fields
                )
        else:
            lchunk_coords, lchunk_selection, lout_selection = zip(*indexer)
            chunk_values = []
            for out_selection in lout_selection:
                if sel_shape == ():
                    chunk_values.append(value)
                elif is_scalar(value, self._dtype):
                    chunk_values.append(value)
                else:
                    chunk_value = value[out_selection]
                    # handle missing singleton dimensions
                    if indexer.drop_axes:  # pragma: no cover
                        item = [slice(None)] * self.ndim
                        for a in indexer.drop_axes:
                            item[a] = np.newaxis
                        item = tuple(item)
                        chunk_value = chunk_value[item]
                    chunk_values.append(chunk_value)

            self._chunk_add(
                lchunk_coords, lchunk_selection, chunk_values, fields=fields
            )

    def _chunk_add(self, chunk_coords, chunk_selection, value, fields=None):
        """Add to part or whole of a chunk.  Borrows heavily from self._chunk_setitem.

        Parameters
        ----------
        chunk_coords : tuple of ints
            Indices of the chunk.
        chunk_selection : tuple of slices
            Location of region within the chunk.
        value : scalar or ndarray
            Value to add.

        """

        if self._synchronizer is None:
            # no synchronization
            lock = nolock
        else:
            # synchronize on the chunk
            ckey = self._chunk_key(chunk_coords)
            lock = self._synchronizer[ckey]

        with lock:
            old = np.empty_like(self._meta_array, shape=self._chunks, dtype=self._dtype)
            old = old[chunk_selection]
            self._chunk_getitems([chunk_coords], [chunk_selection], old, [...])
            self._chunk_setitem_nosync(
                chunk_coords, chunk_selection, old + value, fields=fields
            )
