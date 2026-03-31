"""
Block-based HDF5 data loading for efficient training.

Directly iterating an HDF5 file one sample at a time is very slow because each
read incurs disk-access overhead.  This module solves the problem with a
two-level loading strategy:

  Outer loop (BlockDataLoader / BlockDataset):
      Reads a large contiguous *block* of rows from the HDF5 file into RAM
      (default 10 000 rows).

  Inner loop (ShuffleBlockWrapper + DataLoader):
      Wraps the in-memory block in a standard PyTorch Dataset and yields
      mini-batches of the requested size, shuffled within the block.

Classes
-------
  BlockDataLoader      -- drop-in replacement for torch.utils.data.DataLoader
  HDFDataset           -- thin wrapper around an HDF5 file exposing the
                          (nodes, edges, action probabilities) tensors as a Dataset
  BlockDataset         -- maps block indices → HDF5 row slices
  ShuffleBlockWrapper  -- wraps a preloaded block so the inner DataLoader can
                          shuffle and batch it
"""
# load general packages and functions
from typing import Tuple
import torch
import h5py


class BlockDataLoader(torch.utils.data.DataLoader):
    """
    Two-level DataLoader that reads HDF5 data in large blocks for efficiency.

    Instead of hitting disk once per sample (as a vanilla DataLoader would do
    with an HDF5-backed dataset), ``BlockDataLoader`` reads ``block_size`` rows
    at a time into RAM and then serves ``batch_size`` mini-batches from that
    in-memory block before loading the next one.

    Args:
        dataset:    An ``HDFDataset`` instance.
        batch_size: Number of samples per training mini-batch.
        block_size: Number of HDF5 rows loaded into RAM at once.
        shuffle:    Whether to shuffle blocks and samples within each block.
        n_workers:  Number of worker processes for the outer block loader.
        pin_memory: If True, pin loaded tensors in page-locked memory for
                    faster CPU→GPU transfers (recommended when using CUDA).
    """
    def __init__(self, dataset : torch.utils.data.Dataset, batch_size : int=100,
                block_size : int=10000, shuffle : bool=True, n_workers : int=0,
                pin_memory : bool=True) -> None:

        # define variables to be used throughout dataloading
        self.dataset       = dataset     # `HDFDataset` object
        self.batch_size    = batch_size  # `int`
        self.block_size    = block_size  # `int`
        self.shuffle       = shuffle     # `bool`
        self.n_workers     = n_workers   # `int`
        self.pin_memory    = pin_memory  # `bool`
        self.block_dataset = BlockDataset(self.dataset,
                                          batch_size=self.batch_size,
                                          block_size=self.block_size)

    def __iter__(self) -> torch.Tensor:

        # define a regular `DataLoader` using the `BlockDataset`
        block_loader = torch.utils.data.DataLoader(self.block_dataset,
                                                   shuffle=self.shuffle,
                                                   num_workers=self.n_workers)

        # define a condition for determining whether to drop the last block this
        # is done if the remainder block is very small (less than a tenth the
        # size of a normal block)
        condition = bool(
            int(self.block_dataset.__len__()/self.block_size) > 1 and
            self.block_dataset.__len__()%self.block_size < self.block_size/10
        )

        # loop through and load BLOCKS of data every iteration
        for block in block_loader:
            block = [torch.squeeze(b) for b in block]

            # wrap each block in a `ShuffleBlock` so that data can be shuffled
            # within blocks
            batch_loader = torch.utils.data.DataLoader(
                dataset=ShuffleBlockWrapper(block),
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                pin_memory=self.pin_memory,
                drop_last=condition
            )

            for batch in batch_loader:
                yield batch

    def __len__(self) -> int:
        # returns the number of graphs in the DataLoader
        n_blocks          = len(self.dataset) // self.block_size
        n_rem             = len(self.dataset) % self.block_size
        n_batch_per_block = self.__ceil__(self.block_size, self.batch_size)
        n_last            = self.__ceil__(n_rem, self.batch_size)
        return n_batch_per_block * n_blocks + n_last

    def __ceil__(self, i : int, j : int) -> int:
        return (i + j - 1) // j


class BlockDataset(torch.utils.data.Dataset):
    """
    Modified `Dataset` class which returns BLOCKS of data when `__getitem__()`
    is called.
    """
    def __init__(self, dataset : torch.utils.data.Dataset, batch_size : int=100,
                 block_size : int=10000) -> None:

        assert block_size >= batch_size, "Block size should be > batch size."

        self.block_size = block_size  # `int`
        self.batch_size = batch_size  # `int`
        self.dataset    = dataset     # `HDFDataset`

    def __getitem__(self, idx : int) -> torch.Tensor:
        # returns a block of data from the dataset
        start = idx * self.block_size
        end   = min((idx + 1) * self.block_size, len(self.dataset))
        return self.dataset[start:end]

    def __len__(self) -> int:
        # returns the number of blocks in the dataset
        return (len(self.dataset) + self.block_size - 1) // self.block_size


class ShuffleBlockWrapper:
    """
    Extra class used to wrap a block of data, enabling data to get shuffled
    *within* a block.
    """
    def __init__(self, data : torch.Tensor) -> None:
        self.data = data

    def __getitem__(self, idx : int) -> torch.Tensor:
        return [d[idx] for d in self.data]

    def __len__(self) -> int:
        return len(self.data[0])


class HDFDataset(torch.utils.data.Dataset):
    """
    Reads and collects data from an HDF file with three datasets: "nodes",
    "edges", and "action_probs".
    """
    def __init__(self, path : str) -> None:

        self.path = path
        hdf_file  = h5py.File(self.path, "r+", swmr=True)

        # load each HDF dataset
        self.nodes = hdf_file.get("nodes")
        self.edges = hdf_file.get("edges")
        self.action_probs  = hdf_file.get("action_probs")

        # get the number of elements in the dataset
        self.n_subgraphs = self.nodes.shape[0]

    def __getitem__(self, idx : int) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # returns specific graph elements
        nodes_i = torch.from_numpy(self.nodes[idx]).type(torch.float32)
        edges_i = torch.from_numpy(self.edges[idx]).type(torch.float32)
        action_probs_i   = torch.from_numpy(self.action_probs[idx]).type(torch.float32)

        return (nodes_i, edges_i, action_probs_i)

    def __len__(self) -> int:
        # returns the number of graphs in the dataset
        return self.n_subgraphs
