from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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

