import random

from torch.utils.data import DataLoader, Subset
from util.util import init_obj, set_seed
from loguru import logger
from torch.utils.data.dataloader import Sampler
from functools import partial
import torch
import os


def create_dataloader(options):
    train_dataset_args = options['datasets']['which_dataset']['train_dataset']
    valid_dataset_args = options['datasets']['which_dataset']['valid_dataset']

    generator = torch.Generator()
    generator.manual_seed(options['seed'] + options['local_rank'])

    worker_init_fn = partial(set_seed, gl_seed=options['seed'])

    train_loaders_dict = {}
    valid_loaders_dict = {}

    for single_train_dataset_name, single_train_dataset_args in train_dataset_args.items():
        train_loaders_dict[single_train_dataset_name] = create_single_dataloader(single_train_dataset_args, generator, worker_init_fn, ddp_shuffle=True, options=options)
    logger.success('Train Loader(s) Created.')

    for single_valid_dataset_name, single_valid_dataset_args in valid_dataset_args.items():
        valid_loaders_dict[single_valid_dataset_name] = create_single_dataloader(single_valid_dataset_args, generator, worker_init_fn, ddp_shuffle=False, options=options)
    logger.success('Valid Loader(s) Created.')

    return train_loaders_dict, valid_loaders_dict


def create_single_dataloader(single_dataset_args, generator, worker_init_fn, ddp_shuffle, options):
    dataset = init_obj(single_dataset_args, default_file_name='data.dataset', init_type='Dataset')
    logger.info("Dataset {} has {} samples.".format(dataset.__class__.__name__, len(dataset)))

    dataloader_args = single_dataset_args['loader_args']

    if options['local_rank'] != -1:
        if (not dataloader_args['drop_last']) and ddp_shuffle:
            logger.critical('drop_last == False, which may be conflict with DDP in the training set. PLEASE CHECK THIS!!!')

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=options['local_rank'], shuffle=ddp_shuffle)
        dataloader_args.pop('shuffle', None)
        dataloader = DataLoader(dataset, worker_init_fn=worker_init_fn, generator=generator, sampler=sampler, **dataloader_args)
    else:
        dataloader = DataLoader(dataset, worker_init_fn=worker_init_fn, generator=generator, **dataloader_args)
    logger.success('Loader of {} Created.'.format(dataset.__class__.__name__))

    return dataloader

class ReproducibleSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return ReproducibleIter(self.__len__())

    def __len__(self):
        return len(self.data_source)


class ReproducibleIter:
    def __init__(self, data_count):
        self.data_count = data_count
        self.current_idx = 0
        self.shuffle_list = list(range(self.data_count))
        random.shuffle(self.shuffle_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < self.data_count:
            element = self.shuffle_list[self.current_idx]
            self.current_idx += 1
            return element
        else:
            raise StopIteration

class InfinityIterator:
    def __init__(self, collection):
        assert len(collection) > 0

        self.collection = collection
        self.iter = iter(collection)
        self.epoch = 0

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            if os.environ['LOCAL_RANK'] != '-1':
                self.collection.sampler.set_epoch(self.epoch)
                self.epoch += 1

            self.iter = iter(self.collection)
            return next(self.iter)

    def __len__(self):
        return len(self.collection)
