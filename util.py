from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
from dataloader import get_dataloaders
import time, datetime
from pathlib import Path
import shutil
import logging


'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        config_dir = self.job_dir / 'config.txt'
        #if not os.path.exists(config_dir):
        #if args.resume:
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        '''else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')#'''


def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger




class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def rotrate_concat(inputs, angle_list=[0,90,180,270]):
    out = None
    for x in inputs:
        if 90 in angle_list:
            x_90 = x.transpose(2,3).flip(2)
        else:
            x_90 = torch.Tensor().cuda()

        if 180 in angle_list:
            x_180 = x.flip(2).flip(3)
        else:
            x_180 = torch.Tensor().cuda()

        if 270 in angle_list:  
            x_270 = x.flip(2).transpose(2,3)
        else:
            x_270 = torch.Tensor().cuda()

        if out is None:
            out = torch.cat((x, x_90, x_180, x_270),0)
        else:
            out = torch.cat((out, x, x_90, x_180, x_270),0)

        #print(out.shape)
    return out

    
class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)
        

    def close(self):
        if self.file is not None:
            self.file.close()
            
                   
def generate_final_report(model, opt, wandb):
    from eval.meta_eval import meta_test
    
    opt.n_shots = 1
    train_loader, val_loader, meta_testloader, meta_valloader, _, _ = get_dataloaders(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))
    
    
    wandb.log({'Final Meta Test Acc @1': meta_test_acc,
               'Final Meta Test std @1': meta_test_std,
               'Final Meta Test Acc  (feat) @1': meta_test_acc_feat,
               'Final Meta Test std  (feat) @1': meta_test_std_feat,
               'Final Meta Val Acc @1': meta_val_acc,
               'Final Meta Val std @1': meta_val_std,
               'Final Meta Val Acc   (feat) @1': meta_val_acc_feat,
               'Final Meta Val std   (feat) @1': meta_val_std_feat
              })

    
    opt.n_shots = 5
    train_loader, val_loader, meta_testloader, meta_valloader, _, _ = get_dataloaders(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))

    wandb.log({'Final Meta Test Acc @5': meta_test_acc,
               'Final Meta Test std @5': meta_test_std,
               'Final Meta Test Acc  (feat) @5': meta_test_acc_feat,
               'Final Meta Test std  (feat) @5': meta_test_std_feat,
               'Final Meta Val Acc @5': meta_val_acc,
               'Final Meta Val std @5': meta_val_std,
               'Final Meta Val Acc   (feat) @5': meta_val_acc_feat,
               'Final Meta Val std   (feat) @5': meta_val_std_feat
              })