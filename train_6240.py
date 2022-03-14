from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from tqdm import tqdm
import mkl
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from dataset.transform_cfg import transforms_options, transforms_list
from models import model_pool
from models.util import create_model
from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat, Logger, generate_final_report
from eval.meta_eval import meta_test, meta_test_tune
from eval.cls_eval import validate
import numpy as np
#import wandb
from losses import simple_contrstive_loss
from dataloader import get_dataloaders

from models.resnet_inv_eq import resnet12_1st_half, resnet12_2nd_half
import util

mkl.set_num_threads(2)



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    parser.add_argument('--ssl', type=bool, default=True, help='use self supervised learning')
    parser.add_argument('--tags', type=str, default="gen0, ssl", help='add tags for the experiment')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['cross_domain','miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--job_dir', type=str, default='./result/', help='path to save model')
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='tb/', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='./data/', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    
    #hyper parameters
    parser.add_argument('--gamma', type=float, default=1.0, help='loss cofficient for ssl loss')
    parser.add_argument('--contrast_temp', type=float, default=1.0, help='temperature for contrastive ssl loss')
    parser.add_argument('--membank_size', type=int, default=6400, help='temperature for contrastive ssl loss')
    parser.add_argument('--memfeature_size', type=int, default=64, help='temperature for contrastive ssl loss')
    parser.add_argument('--mvavg_rate', type=float, default=0.99, help='temperature for contrastive ssl loss')
    parser.add_argument('--trans', type=int, default=16, help='number of transformations')
    
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='Select gpu to use')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    util.record_config(args)
    logger = util.get_logger(os.path.join(args.job_dir, 'logger.log'))

    if args.dataset == 'CIFAR-FS' or args.dataset == 'FC100':
        args.transform = 'D'

    if args.use_trainval:
        args.trial = args.trial + '_trainval'

    # set the path according to the environment
    #if not args.model_path:
    #    args.model_path = './models_pretrained'
    #if not args.tb_path:
    #    args.tb_path = './tensorboard'
    #if not args.data_root:
    #    args.data_root = './data/{}'.format(args.dataset)
    #else:
    #    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    args.data_aug = True

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
        
    tags = args.tags.split(',')
    args.tags = list([])
    for it in tags:
        args.tags.append(it)

    args.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(args.model, args.dataset, args.learning_rate, args.weight_decay, args.transform)

    if args.cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    if args.adam:
        args.model_name = '{}_useAdam'.format(args.model_name)

    args.model_name = '{}_trial_{}'.format(args.model_name, args.trial)

    #args.tb_folder = os.path.join(args.tb_path, args.model_name)
    ##if not os.path.isdir(args.tb_folder):
    #    os.makedirs(args.tb_folder)

    #args.save_folder = os.path.join(args.model_path, args.model_name)
    #if not os.path.isdir(args.save_folder):
    #    os.makedirs(args.save_folder)

    args.n_gpu = torch.cuda.device_count()

    #extras
    args.fresh_start = True
    return args



class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class MMClassifier_domain(nn.Module):
    
    def __init__(self, n_class, lamda=1.):
        super(MMClassifier_domain, self).__init__()
        self.lamda=lamda
        self.grad_reverse = GradientReversal(lambda_=self.lamda)
        
        self.conv1 = nn.Conv2d(160, 128, 3, 1)
        #self.conv2 = nn.Conv2d(128, 64, 3, 1)
        self.fc1 = nn.Linear(128 * 9 * 9, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)
        
        #self.fc1 = nn.Linear(1280, 256)
        #self.fc2 = nn.Linear(256, n_class)
        #self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        #x = x.mean(3).mean(2)
        x = self.grad_reverse(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class splitlow_commonhigh(nn.Module):
    def __init__(self, args, n_cls):
        super(splitlow_commonhigh, self).__init__()

        self.class_numbers = n_cls
        self.args = args
        self.batch_size = args.batch_size

        # model define
        self.model_p1 = eval(args.model+'_1st_half')(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls, no_trans=args.trans, embd_size=args.memfeature_size)
        self.model_p2 = eval(args.model+'_1st_half')(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls, no_trans=args.trans, embd_size=args.memfeature_size)
        self.model_c = eval(args.model+'_2nd_half')(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls, no_trans=args.trans, embd_size=args.memfeature_size)
        self.model_d = MMClassifier_domain(n_class=2).cuda()
        #model_clf
        
        '''pretrain_dir=args.pretrain_dir
        if pretrain_dir:
            from collections import OrderedDict
            ckpt = torch.load(pretrain_dir)
            print('resuming pretrain model from %s' % pretrain_dir)
            new_state_dict = OrderedDict()
            for k, v in ckpt['student'].items():
                new_state_dict[k.replace('module.','')] = v
            self.model_p1.load_state_dict(new_state_dict, strict=False)
            self.model_p2.load_state_dict(new_state_dict, strict=False)
            self.model_c.load_state_dict(new_state_dict, strict=False)#'''

        if len(args.gpu) > 1:
            raise NotImplementedError
            device_id = []
            for i in range((len(args.gpu) + 1) // 2):
                device_id.append(i)
            model = nn.DataParallel(model, device_ids=device_id).cuda()
        else:
            self.model_p1.cuda()
            self.model_p2.cuda()
            self.model_c.cuda()
            self.model_d.cuda()

        #logger.info('-------model private:---------')
        #logger.info(model_p1)
        #logger.info('-------model common:---------')
        #logger.info(model_c)


    def forward(self, images_s, images_t=None, is_training=False, inductive=False):
        if is_training:
            f0_s,f1_s = self.model_p1(images_s)
            f0_t,f1_t = self.model_p2(images_t)
            mid_features = torch.cat([f1_s, f1_t], dim=0)
            
            pred, conv_feat, eq_logit, inv_logit = self.model_c(mid_features, inductive=inductive)
            pred_s, pred_t = pred[:self.batch_size], pred[self.batch_size:]
            conv_feat_s, conv_feat_t = conv_feat[:self.batch_size], conv_feat[self.batch_size:]
            eq_logit_s, eq_logit_t = eq_logit[:self.batch_size], eq_logit[self.batch_size:]
            inv_logit_s, inv_logit_t = inv_logit[:self.batch_size], inv_logit[self.batch_size:]
            
            pred_domain = self.model_d(mid_features)
            pred_domain_s, pred_domain_t = pred_domain[:self.args.batch_size], pred_domain[self.args.batch_size:]
            
            return (pred_s, pred_t), (conv_feat_s, conv_feat_t), (eq_logit_s, eq_logit_t), (inv_logit_s, inv_logit_t), (pred_domain_s, pred_domain_t)
        else:
            _,f = self.model_p1(images_s)
            pred_s,_,_,feature_s,_ = self.model_c(f, is_training=False, batch_size=self.args.batch_size)
            return pred_s, feature_s




def main():

    args = parse_option()
    #wandb.init(project=args.model_path.split("/")[-1], tags=args.tags)
    #wandb.config.update(args)
    #wandb.save('*.py')
    #wandb.run.save()
       
    train_loader, val_loader, meta_testloader, meta_valloader, n_cls, no_sample = get_dataloaders(args)
    # model
    #model = create_model(args.model, n_cls, args.dataset, n_trans=args.trans, embd_sz=args.memfeature_size)
    model = splitlow_commonhigh(args, n_cls)
    #wandb.watch(model)
    
    # optimizer
    if args.adam:
        print("Adam")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=0.0005)
    else:
        print("SGD")
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    '''if torch.cuda.is_available():
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True#'''

    # set cosine annealing scheduler
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)
    
    MemBank = np.random.randn(no_sample, args.memfeature_size)
    MemBank = torch.tensor(MemBank, dtype=torch.float).cuda()
    MemBankNorm = torch.norm(MemBank, dim=1, keepdim=True)
    MemBank = MemBank / (MemBankNorm + 1e-6)

    # routine: supervised pre-training
    for epoch in range(1, args.epochs + 1):
        if args.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        
        time1 = time.time()
        train_acc, train_loss, MemBank = train(epoch, train_loader, model, criterion, optimizer,args, MemBank)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        val_acc, val_acc_top5, val_loss = 0,0,0 #validate(val_loader, model, criterion, args)
        
        #validate
        start = time.time()
        meta_val_acc, meta_val_std = 0,0 #meta_test(model, meta_valloader)
        test_time = time.time() - start
        print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}, Time: {:.1f}'.format(meta_val_acc, meta_val_std, test_time))

        #evaluate
        start = time.time()
        meta_test_acc, meta_test_std = 0,0 #meta_test(model, meta_testloader)
        test_time = time.time() - start
        print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}, Time: {:.1f}'.format(meta_test_acc, meta_test_std, test_time))
        
        # regular saving
        if epoch % args.save_freq == 0 or epoch==args.epochs:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }  
            save_dir = os.path.join(args.job_dir, 'model/')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, 'model_%d.pth'%(epoch))
            torch.save(state, filename)
            
            #wandb saving
            #torch.save(state, os.path.join(args.job_dir, "model.pth"))

        '''wandb.log({'epoch': epoch, 
                   'Train Acc': train_acc,
                   'Train Loss':train_loss,
                   'Val Acc': val_acc,
                   'Val Loss':val_loss,
                   'Meta Test Acc': meta_test_acc,
                   'Meta Test std': meta_test_std,
                   'Meta Val Acc': meta_val_acc,
                   'Meta Val std': meta_val_std
                  })#'''

    #final report 
    #print("GENERATING FINAL REPORT")
    #generate_final_report(model, opt, wandb)
    
    #remove output.txt log file 
    output_log_file = os.path.join(args.job_dir, "output.log")
    if os.path.isfile(output_log_file):
        os.remove(output_log_file)
    else:    ## Show an error ##
        print("Error: %s file not found" % output_log_file)
        
      
def train(epoch, train_loader, model, criterion, optimizer, args, MemBank):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_indices = list(range(len(MemBank)))

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        #for _, (input, input2, input3, input4, target, indices) in enumerate(pbar):
        for _, ((image_s, target_s, indices_s), (image_t, _, indices_t)) in enumerate(pbar):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                image_s, target_s, indices_s, image_t, indices_t= image_s.cuda(), target_s.cuda(), indices_s.cuda(), image_t.cuda(), indices_t.cuda()
                #input = input.cuda()
                #input2 = input2.cuda()
                #input3 = input3.cuda()
                #input4 = input4.cuda()
                #target = target.cuda()
                #indices = indices.cuda()
            batch_size = image_s.shape[0]

            #generated_data = rotrate_concat([input, input2, input3, input4])
            #train_targets = target.repeat(args.trans)
            #proxy_labels = torch.zeros(args.trans*batch_size).cuda().long()

            #for ii in range(args.trans):
            #    proxy_labels[ii*batch_size:(ii+1)*batch_size] = ii

            # ===================forward=====================
            #_, (train_logit, eq_logit, inv_rep) = model(input, inductive=True)
            (pred_s, pred_t), (conv_feat_s, conv_feat_t), (eq_logit_s, eq_logit_t), (inv_logit_s, inv_logit_t), (pred_domain_s, pred_domain_t) = model(image_s, image_t, is_training=True, inductive=True)

            # ===================memory bank of negatives for current batch=====================
            np.random.shuffle(train_indices)
            mn_indices_all = np.array(list(set(train_indices) - set(indices_s)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:args.membank_size]
            mn_arr = MemBank[mn_indices]
            mem_rep_of_batch_imgs = MemBank[indices_s]#'''

            loss_ce = criterion(pred_s, target_s)
            #loss_eq = criterion(eq_logit_s, proxy_labels)

            inv_rep_0 = inv_logit_s[:batch_size, :]
            loss_inv = simple_contrstive_loss(mem_rep_of_batch_imgs, inv_rep_0, mn_arr, args.contrast_temp)
            for ii in range(1, args.trans):
                loss_inv += simple_contrstive_loss(inv_rep_0, inv_logit_s[(ii*batch_size):((ii+1)*batch_size), :], mn_arr, args.contrast_temp)
            loss_inv = loss_inv/args.trans#'''

            #loss = args.gamma * (loss_eq + loss_inv) + loss_ce
            loss = args.gamma * (loss_inv) + loss_ce
            
            n=image_s.size(0)
            acc1, acc5 = accuracy(pred_s, target_s, topk=(1, 5))
            losses.update(loss.item(), n)
            top1.update(acc1[0], n)
            top5.update(acc5[0], n)

            # ===================update memory bank======================
            MemBankCopy = MemBank.clone().detach()
            MemBankCopy[indices_s] = (args.mvavg_rate * MemBankCopy[indices_s]) + ((1 - args.mvavg_rate) * inv_rep_0)
            MemBank = MemBankCopy.clone().detach()#'''

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                              "Acc@5":'{0:.2f}'.format(top5.avg.cpu().numpy(),2), 
                              "Loss" :'{0:.2f}'.format(losses.avg,2),
                             })

    print('Train_Acc@1 {top1.avg:.3f} Train_Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, losses.avg, MemBank


if __name__ == '__main__':
    main()