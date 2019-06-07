import os
import time
import numpy as np
from torchstat import stat
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from processbar import ProcessBar
from utils import getTime, flip, distCosine, cvSelectThreshold


class Trainer(object):
    """ Train Templet
    """

    def __init__(self, configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1):

        self.configer = configer
        self.valid_freq = valid_freq

        self.net = net
        
        ## directory for log and checkpoints
        self.logdir = os.path.join(configer.logdir, self.net._get_name())
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = configer.ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        
        ## datasets
        self.trainset = trainset
        self.validset = validset
        self.trainloader = DataLoader(trainset, configer.batchsize, True)
        self.validloader = DataLoader(validset, configer.batchsize // 4, True)

        ## for optimization
        self.criterion = criterion
        self.optimizer = optimizer(params, configer.lrbase, momentum=0.9, nesterov=True)
        self.lr_scheduler = lr_scheduler(self.optimizer, configer.adjstep, configer.gamma)
        self.writer = SummaryWriter(configer.logdir)
        # self.writer.add_graph(self.net, (torch.rand([1] + configer.inputsize), ))
        
        ## initialize
        self.valid_loss = float('inf')
        self.elapsed_time = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        self.save_times = 0
        self.num_to_keep = num_to_keep

        ## if resume
        if resume:
            self.load_checkpoint()

        stat(self.net, configer.inputsize)
        if configer.cuda and cuda.is_available(): 
            self.net.cuda()

        print("==============================================================================================")
        print("model:           {}".format(self.net._get_name()))
        print("logdir:          {}".format(self.logdir))
        print("ckptdir:         {}".format(self.ckptdir))
        print("train samples:   {}k".format(len(trainset)/1000))
        print("valid samples:   {}k".format(len(validset)/1000))
        print("batch size:      {}".format(configer.batchsize))
        print("batch per epoch: {}".format(len(trainset)/configer.batchsize))
        print("epoch:           [{:4d}]/[{:4d}]".format(self.cur_epoch, configer.n_epoch))
        print("learing rate:    {}".format(configer.lrbase))
        print("==============================================================================================")

    def train(self):
        
        n_epoch = self.configer.n_epoch - self.cur_epoch
        print("Start training! current epoch: {}, remain epoch: {}".format(self.cur_epoch, n_epoch))

        bar = ProcessBar(n_epoch)
        loss_train = 0.; loss_valid = 0.

        for i_epoch in range(n_epoch):
            
            if self.configer.cuda and cuda.is_available(): cuda.empty_cache()

            self.cur_epoch += 1
            bar.step()

            self.lr_scheduler.step(self.cur_epoch)
            cur_lr = self.lr_scheduler.get_lr()[-1]
            self.writer.add_scalar('{}/lr'.format(self.net._get_name()), cur_lr, self.cur_epoch)

            loss_train = self.train_epoch()
            # print("----------------------------------------------------------------------------------------------")
            if self.valid_freq != 0 and self.cur_epoch % self.valid_freq == 0:
                loss_valid = self.valid_epoch()
            # print("----------------------------------------------------------------------------------------------")

            self.writer.add_scalars('loss', {'train': loss_train, 'valid': loss_valid}, self.cur_epoch)

            # print_log = "{} || Elapsed: {:.4f}h || Epoch: [{:3d}]/[{:3d}] || lr: {:.6f},| train loss: {:4.4f}, valid loss: {:4.4f}".\
            #         format(getTime(), self.elapsed_time/3600, self.cur_epoch, self.configer.n_epoch, 
            #             cur_lr, loss_train, loss_valid)
            # print(print_log)

            if self.valid_freq == 0:
                self.save_checkpoint()
            
            else:
                if loss_valid < self.valid_loss:
                    self.valid_loss = loss_valid
                    self.save_checkpoint()
                
            # print("==============================================================================================")


    def train_epoch(self):
        
        self.net.train()
        avg_loss = []
        start_time = time.time()
        n_batch = len(self.trainset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.trainloader):

            self.cur_batch += 1

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            y_pred = self.net(X)
            loss_i = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            avg_loss += [loss_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/train/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()
            self.elapsed_time += duration_time
            total_time = duration_time * self.configer.n_epoch * len(self.trainset) // self.configer.batchsize
            left_time = total_time - self.elapsed_time

            # print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] | cur: [{:3d}] || lr: {:.6f}, loss: {:4.4f}".\
            #     format(getTime(), self.elapsed_time/3600, left_time/3600, self.configer.batchsize / duration_time,
            #         self.cur_epoch, self.configer.n_epoch, i_batch, n_batch, self.cur_batch,
            #         self.lr_scheduler.get_lr()[-1], loss_i
            #     )
            # print(print_log)
        
        avg_loss = np.mean(np.array(avg_loss))
        return avg_loss


    def valid_epoch(self):
        
        self.net.eval()
        avg_loss = []
        start_time = time.time()
        n_batch = len(self.validset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.validloader):

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            y_pred = self.net(X)
            loss_i = self.criterion(y_pred, y)

            avg_loss += [loss_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/valid/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()

            # print_log = "{} || FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || loss: {:4.4f}".\
            #     format(getTime(), self.configer.batchsize / duration_time,
            #         self.cur_epoch, self.configer.n_epoch, i_batch, n_batch, loss_i
            #     )
            # print(print_log)
        
        avg_loss = np.mean(np.array(avg_loss))
        return avg_loss
    
    def save_checkpoint(self):
        
        checkpoint_state = {
            'save_time': getTime(),

            'cur_epoch': self.cur_epoch,
            'cur_batch': self.cur_batch,
            'elapsed_time': self.elapsed_time,
            'valid_loss': self.valid_loss,
            'save_times': self.save_times,
            
            'net': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times))
        torch.save(checkpoint_state, checkpoint_path)
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times-self.num_to_keep))
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

        self.save_times += 1

        # print("checkpoint saved at {}".format(checkpoint_path))

    def load_checkpoint(self, index):
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), index))
        checkpoint_state = torch.load(checkpoint_path, map_location='cuda' if cuda.is_available() else 'cpu')
        
        self.cur_epoch = checkpoint_state['cur_epoch']
        self.cur_batch = checkpoint_state['cur_batch']
        self.elapsed_time = checkpoint_state['elapsed_time']
        self.valid_loss = checkpoint_state['valid_loss']
        self.save_times = checkpoint_state['save_times']

        self.net.load_state_dict(checkpoint_state['net'])
        self.optimizer.load_state_dict(checkpoint_state['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler_state'])

        # print("load checkpoint from {}, last save time: {}".\
        #                         format(checkpoint_path, checkpoint_state['save_time']))





class MobileFacenetTrainer():

    def __init__(self, configer, net, params, classifyData, verifyData, criterion,
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1):

        self.configer = configer
        self.valid_freq = valid_freq
        self.net = net
        
        ## directory for log and checkpoints
        self.logdir = os.path.join(configer.logdir, self.net._get_name())
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = configer.ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        
        ## datasets
        self.classifyData = classifyData
        self.verifyData   = verifyData
        self.classifyLoader = DataLoader(classifyData, configer.batchsize, True)
        self.verifyLoader   = DataLoader(verifyData,   1, False)

        ## for optimization
        self.criterion = criterion
        self.optimizer = optimizer(params, configer.lrbase, momentum=0.9, nesterov=True)
        self.lr_scheduler = lr_scheduler(self.optimizer, configer.adjstep, configer.gamma)
        self.writer = SummaryWriter(configer.logdir)
        
        ## initialize
        self.threshBest = 0.
        self.accBest = float('-inf')
        self.f1Best  = float('-inf')
        
        self.cur_epoch = 0
        self.cur_batch = 0
        self.save_times = 0
        self.num_to_keep = num_to_keep
        self.thresh = 0

        ## if resume
        if resume:
            self.load_checkpoint()

        if configer.cuda and cuda.is_available(): 
            self.net.cuda()

        print("==============================================================================================")
        print("model:           {}".format(self.net._get_name()))
        print("logdir:          {}".format(self.logdir))
        print("ckptdir:         {}".format(self.ckptdir))
        print("classify samples:{}k".format(len(classifyData)/1000))
        print("verify samples:  {}k".format(len(verifyData)/1000))
        print("batch size:      {}".format(configer.batchsize))
        print("batch per epoch: {}".format(len(classifyData)/configer.batchsize))
        print("epoch:           [{:4d}]/[{:4d}]".format(self.cur_epoch, configer.n_epoch))
        print("learing rate:    {}".format(configer.lrbase))
        print("==============================================================================================")

    def train(self):

        n_epoch = self.configer.n_epoch - self.cur_epoch
        print("Start training! current epoch: {}, remain epoch: {}".format(self.cur_epoch, n_epoch))
        
        bar = ProcessBar(n_epoch)

        for i_epoch in range(n_epoch):

            if self.configer.cuda and cuda.is_available():
                cuda.empty_cache()

            self.cur_epoch += 1
            bar.step()        

            self.lr_scheduler.step(self.cur_epoch)
            cur_lr = self.lr_scheduler.get_lr()[-1]
            self.writer.add_scalar('{}/lr'.format(self.net._get_name()), cur_lr, self.cur_epoch)

            loss_train = self.train_epoch()
            if self.valid_freq != 0 and self.cur_epoch % self.valid_freq == 0:
                thresh, acc, f1 = self.valid_epoch()

            self.writer.add_scalars('{}/valid/'.format(self.net._get_name()), 
                        {'threshold': thresh, 'accuracy': acc, 'f1score': f1}, self.cur_epoch)

            if self.valid_freq == 0:
                self.save_checkpoint()
            
            else:
                if f1 > self.f1Best:
                    self.threshBest = thresh
                    self.accBest = acc
                    self.f1Best = f1
                    self.save_checkpoint()


    def train_epoch(self):

        self.net.train(); avg_loss = []
        n_batch = len(self.classifyData) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.classifyLoader):

            self.cur_batch += 1

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available():
                X = X.cuda(); y = y.cuda()

            y_pred = self.net(X)
            loss_i = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            avg_loss += [loss_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/train/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)

        avg_loss = np.mean(np.array(avg_loss))
        return avg_loss

    def valid_epoch(self):

        self.net.eval(); gt = []; pred = []
        for i, (X1, X2, y_true) in enumerate(self.verifyLoader):

            if cuda.is_available():
                X1, X2, y_true = X1.cuda(), X2.cuda(), y_true.cuda()

            feat1 = torch.cat([self.net.get_feature(X1), self.net.get_feature(flip(X1))], dim=1).view(-1)
            feat2 = torch.cat([self.net.get_feature(X2), self.net.get_feature(flip(X2))], dim=1).view(-1)
            cosine = distCosine(feat1, feat2)

            gt += [y_true]; pred += [cosine.detach()]

        gt   = torch.cat(list(map(lambda x: x             ,   gt)), dim=0).cpu().numpy()
        pred = torch.cat(list(map(lambda x: x.unsqueeze(0), pred)), dim=0).cpu().numpy()
        thresh, acc, f1 = cvSelectThreshold(pred, gt)

        return thresh, acc, f1

    def save_checkpoint(self):
        
        checkpoint_state = {
            'save_time': getTime(),

            'cur_epoch': self.cur_epoch,
            'cur_batch': self.cur_batch,
            'threshold': self.threshBest,
            'accuracy':  self.accBest,
            'f1score':   self.f1Best,

            'save_times': self.save_times,

            'net': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times))
        torch.save(checkpoint_state, checkpoint_path)
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times-self.num_to_keep))
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

        self.save_times += 1
        
    def load_checkpoint(self, index):
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), index))
        checkpoint_state = torch.load(checkpoint_path, map_location='cuda' \
                            if self.configer.cuda and cuda.is_available() else 'cpu')
        
        self.cur_epoch = checkpoint_state['cur_epoch']
        self.cur_batch = checkpoint_state['cur_batch']
        self.elapsed_time = checkpoint_state['elapsed_time']
        self.threshBest = checkpoint_state['threshold']
        self.accBest = checkpoint_state['accuracy']
        self.f1Best = checkpoint_state['f1score']
        self.save_times = checkpoint_state['save_times']

        self.net.load_state_dict(checkpoint_state['net'])
        self.optimizer.load_state_dict(checkpoint_state['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler_state'])






class MobileFacenetUnsupervisedTrainer():

    def __init__(self):

        pass

    def train(self):

        pass

    def train_epoch(self):

        pass
    
    def valid_epoch(self):

        pass

    def save_checkpoint(self):
        
        pass

    def load_checkpoint(self, index):
        
        pass
