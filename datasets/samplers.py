from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(dataset)

        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        if num_samples is None:
            self.num_samples = len(self.indices)
        else:
            self.num_samples = num_samples

        target_counts = self.get_target_counts(dataset)

        weights = []
        for idx in self.indices:
            target = self.get_target(dataset, idx)
            weights += [1.0 / target_counts[target]]

        self.weights = torch.Tensor(weights).float()

    def __iter__(self):
        return (
            self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples

    @staticmethod
    def get_target_counts(dataset: Dataset):
        if dataset.__class__.__name__ == 'WM811K':
            targets = [s[-1] for s in dataset.samples]
        else:
            targets = dataset.targets
        return Counter(targets)

    @staticmethod
    def get_target(dataset: Dataset, idx: int):
        if dataset.__class__.__name__ == 'WM811K':
            return dataset.samples[idx][-1]

