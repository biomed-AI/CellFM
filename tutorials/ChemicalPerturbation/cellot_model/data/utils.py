from cellot_model.utils.helpers import nest_dict, flat_dict
from torch.utils.data import DataLoader, Dataset
from itertools import groupby
from absl import logging


def cast_dataset_to_loader(dataset, **kwargs):
    # check if dataset is torch.utils.data.Dataset
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, **kwargs)

    batch_size = kwargs.pop('batch_size', 1)
    kwargs.pop('shuffle')
    flat_dataset = flat_dict(dataset)

    # for key in flat_dataset:
    #     print(key, len(flat_dataset[key]))

    minimum_batch_size = {
        group: min(*map(lambda x: len(flat_dataset[x]), keys), batch_size)
        for group, keys
        in groupby(flat_dataset.keys(), key=lambda x: x.split('.')[0])
    }

    min_bs = min(minimum_batch_size.values())
    if batch_size != min_bs:
        logging.warn(f'Batch size adapted to {min_bs} due to dataset size.')

    loader = nest_dict({
        key: DataLoader(
            val,
            batch_size= len(val) if key.split('.')[0] in ['test', 'ood'] else minimum_batch_size[key.split('.')[0]],
            shuffle=False if key.split('.')[0] in ['test', 'ood'] else True,
            **kwargs)
        for key, val
        in flat_dataset.items()
    }, as_dot_dict=True)

    return loader


def cast_loader_to_iterator(loader, cycle_all=True):
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    if isinstance(loader, DataLoader):
        return cycle(loader)

    iterator = nest_dict({
        key: cycle(item)
        for key, item
        in flat_dict(loader).items()
    }, as_dot_dict=True)

    for value in flat_dict(loader).values():
        assert len(value) > 0

    return iterator
