import grain.python as grain
import h5py
import jax
import numpy as np

class HDF5DataSource(grain.RandomAccessDataSource):
    def __init__(self, path):
        self._path = path
        with h5py.File(path, "r") as f:
            self._len = f["sequences"].shape[0]
            self._has_labels = "labels" in f
        # keep file closed; open per-read for thread safety
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        with h5py.File(self._path, "r") as f:
            item = {"sequences": f["sequences"][idx]}
            if self._has_labels:
                item["labels"] = f["labels"][idx]
        return item


class HDF5BatchLoader:
    """Mini-batch loader for integer-encoded HDF5 datasets.

    The HDF5 file must contain a `sequences` dataset (X). If a `labels` dataset
    is present, this loader yields `(X, y)` tuples. Otherwise it yields `X` only.
    """

    def __init__(
        self,
        path,
        batch_size,
        shuffle=True,
        drop_last=False,
        seed=0,
        to_device=False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.path = path
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.to_device = bool(to_device)

        self._source = HDF5DataSource(path)
        self._size = len(self._source)
        self._has_labels = self._source._has_labels
        self._rng = np.random.default_rng(seed)
        self._cpu_device = jax.devices("cpu")[0]

    def _to_jax_array(self, array):
        if self.to_device:
            return jax.device_put(array)
        return jax.device_put(array, device=self._cpu_device)

    @property
    def has_labels(self):
        return self._has_labels

    @property
    def size(self):
        return self._size

    def __len__(self):
        if self.drop_last:
            return self._size // self.batch_size
        return (self._size + self.batch_size - 1) // self.batch_size

    def _epoch_indices(self):
        indices = np.arange(self._size)
        if self.shuffle:
            self._rng.shuffle(indices)
        return indices

    def __iter__(self):
        stop = self._size if not self.drop_last else (self._size // self.batch_size) * self.batch_size

        # Keep one file handle open for the whole epoch to avoid per-sample I/O.
        with h5py.File(self.path, "r") as f:
            sequences_ds = f["sequences"]
            labels_ds = f["labels"] if self._has_labels else None

            # Fast path: contiguous reads are significantly cheaper than
            # shuffled fancy indexing in HDF5.
            if not self.shuffle:
                for start in range(0, stop, self.batch_size):
                    end = start + self.batch_size
                    x_np = sequences_ds[start:end]
                    x = self._to_jax_array(x_np)

                    if self._has_labels:
                        y_np = labels_ds[start:end]
                        y = self._to_jax_array(y_np)
                        yield x, y
                    else:
                        yield x
                return

            indices = self._epoch_indices()

            for start in range(0, stop, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                # h5py fancy indexing expects monotonically increasing indices.
                sort_order = np.argsort(batch_idx)
                sorted_idx = batch_idx[sort_order]
                inverse_order = np.empty_like(sort_order)
                inverse_order[sort_order] = np.arange(sort_order.shape[0])

                x_sorted = sequences_ds[sorted_idx]
                x_np = x_sorted[inverse_order]
                x = self._to_jax_array(x_np)

                if self._has_labels:
                    y_sorted = labels_ds[sorted_idx]
                    y_np = y_sorted[inverse_order]
                    y = self._to_jax_array(y_np)
                    yield x, y
                else:
                    yield x


def create_dataloader(
    path,
    batch_size,
    shuffle=True,
    drop_last=False,
    seed=0,
    to_device=False,
):
    """Create an iterable loader that returns JAX batches from an HDF5 file.

    Examples:
        loader = create_dataloader("project/step_1_data_preparation/chembl_test_aug.h5", 64)
        for x in loader:
            ...

        loader = create_dataloader("project/step_1_data_preparation/chembl224_ki_test_aug.h5", 64)
        for x, y in loader:
            ...
    """

    return HDF5BatchLoader(
        path=path,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        to_device=to_device,
    )

