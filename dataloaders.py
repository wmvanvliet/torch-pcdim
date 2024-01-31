"""
PyTorch dataloaders to load the weird dataformat I'm using for my datasets. In
order to conserve memory, my datasets are pickled lists of PNG encoded images.
Reading these datasets involves using the Python Imaging Library (PIL) to
decode the PNG binary strings into PIL images.

The CombinedPickledPNGs dataloader will concatenate multiple datasets together.
This is useful is you want to train on for example both imagenet and word
datasets.
"""
import os.path as op
import pandas as pd
from glob import glob

import torch
from torch.utils.data import IterableDataset
import numpy as np
import webdataset as wds


class WebDataset(IterableDataset):
    """Reads datasets in webdataset form

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        shuffle (bool, optional): If True, produces the items in randomized
            order.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        labels ('int' | 'one-hot' | 'vector'): What kind of labels to use. Either a integer
            class label, or a one-hot encoded class label, or a distributed (word2vec) vector.
        label_offset (int): offset for 'int' style labels
    """
    def __init__(self, root, train=True, shuffle=True, transform=None,
                 target_transform=None, labels='int'):
        super().__init__()
        self.shuffle = shuffle
        self.transform = transform
        if labels not in ['int', 'one-hot', 'vector']:
            raise ValueError(f'Invalid label type {labels}, needs to be either "int" or "vector"')
        self.labels = labels
        base_fname = 'train' if train else 'test'
        self.meta = pd.read_csv(op.join(root, f'{base_fname}.csv'), index_col=0)
        #self.vectors = np.atleast_2d(np.loadtxt(op.join(root, 'vectors.csv'), delimiter=',', skiprows=1, usecols=np.arange(1, 301), encoding='utf8', dtype=np.float32, comments=None))
        self.classes = self.meta.groupby('label').agg('first')['text']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        if target_transform is not None:
            self.target_transform = target_transform
        elif labels == 'int':
            self.target_transform = lambda x: int(x)
        elif labels == 'one-hot':
            self.one_hot = torch.eye(len(self.classes))
            self.target_transform = lambda x: self.one_hot[x]
        elif labels == 'vector':
            self.target_transform = lambda x: (int(x), self.vectors[x])

        self.dataset = wds.WebDataset(glob(f'{root}/{base_fname}/*.tar'), shardshuffle=shuffle)
        if shuffle:
            self.dataset = self.dataset.shuffle(1000)
        self.data = self.dataset.decode('pil').to_tuple('png;jpg;jpeg cls').map_tuple(transform, self.target_transform)
        self.iter = iter(self.data)
        self.length = len(self.meta)

    def batched(self, batch_size):
        self.data = self.data.batched(batch_size)
        self.iter = iter(self.data)
        self.length //= batch_size
        return self

    def __next__(self):
        return next(self.iter)

    def __iter__(self):
        """
        Yields:
            tuple: (image, target) where target is index of the target class.
        """
        self.iter = iter(self.data)
        return self.iter

    def __len__(self):
        return self.length


class Combined(IterableDataset):
    def __init__(self, datasets, interleave=True, batch_size=None):
        super().__init__()
        self.datasets = datasets
        self.interleave = interleave
        self.batch_size=batch_size

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        if self.interleave:
            n = np.array([len(ds) for ds in self.datasets])
            repeats = np.round(np.max(n) / n).astype(int)
            if self.batch_size is None:
                for i in range(np.max(n)):
                    for repeat_every, source in zip(repeats, sources):
                        if i % repeat_every == 0:
                            try:
                                yield next(source)
                            except StopIteration:
                                continue
            else:
                self.batch_sizes = np.round(n / np.sum(n) * self.batch_size).astype(int)
                sources = [iter(ds.batched(batch_size)) for ds, batch_size in zip(self.datasets, self.batch_sizes)]
                for i in range(max(len(s) for s in sources)):
                    Xs = []
                    ys = []
                    for source in sources:
                        try:
                            X, y = next(source)
                            Xs.append(X)
                            ys.append(y)
                        except StopIteration:
                            continue
                    if len(Xs) == 0 or len(ys) == 0:
                        return
                    ys = np.concatenate(ys, axis=0)
                    if isinstance(Xs[0], torch.Tensor):
                        Xs = torch.concat(Xs, dim=0)
                    elif isinstance(Xs[0], np.ndarray):
                        Xs = np.concatenate(Xs, axis=0)
                    else:
                        Xs = sum(Xs, [])
                    yield Xs, ys

        else:
            for source in sources:
                if self.batch_size is not None:
                    yield from source.batched(self.batch_size)
                else:
                    yield from source

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])
