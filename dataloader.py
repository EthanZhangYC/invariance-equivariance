from __future__ import print_function

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.CUB import CUB, MetaCUB
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, CIFAR100_6240, MetaCIFAR100, CIFAR100_toy
from dataset.transform_cfg import transforms_options, transforms_test_options, transforms_list


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.dataset0=self.datasets[0]
        self.dataset1=self.datasets[1]

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    #def __getitem__(self, i):
    #    return tuple(d[i %len(d)] for d in self.datasets)

    #def __len__(self):
    #    return max(len(d) for d in self.datasets)

   
def get_dataloaders(opt):
    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    
    if opt.dataset == 'cross_domain':

        train_trans, test_trans = transforms_options[opt.transform]
        
        cub_trainset = CUB(args=opt, partition=train_partition, transform=train_trans)
        miniImgnet_trainset = ImageNet(args=opt, partition=train_partition, transform=train_trans)
        trainset_concat =  ConcatDataset(miniImgnet_trainset, cub_trainset )
        
        train_loader = DataLoader(trainset_concat,
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        

        val_loader = DataLoader(CUB(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
        
        no_sample = len(ImageNet(args=opt, partition=train_partition, transform=train_trans))
        
    elif opt.dataset == 'miniImageNet':

        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
        
        no_sample = len(ImageNet(args=opt, partition=train_partition, transform=train_trans))

    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
        
        no_sample = len(TieredImageNet(args=opt, partition=train_partition, transform=train_trans))

    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        
        meta_trainloader = DataLoader(MetaCIFAR100(args=opt, partition='train',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=1, shuffle=True, drop_last=False,
                                     num_workers=opt.num_workers)
        
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val', 
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
        no_sample = len(CIFAR100(args=opt, partition=train_partition, transform=train_trans))      
    else:
        raise NotImplementedError(opt.dataset)
        
    return train_loader, val_loader, meta_testloader, meta_valloader, n_cls, no_sample