#!/usr/bin/env python

from allennlp.data import PyTorchDataLoader, DatasetReader, Vocabulary
from allennlp.common.params import Params
from allennlp.common.plugins import import_plugins
import torch.multiprocessing as mp


def main():
    mp.spawn(worker_fn, nprocs=1)
    print("done!")


def worker_fn(rank=None):
    import_plugins()

    reader = DatasetReader.from_params(Params({"type": "my_reader", "lazy": True}))
    vocab = Vocabulary.empty()

    dataset = reader.read("path")
    dataset.index_with(vocab)

    loader = PyTorchDataLoader(dataset, num_workers=1)
    list(loader)


if __name__ == "__main__":
    main()
