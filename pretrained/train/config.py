from easydict import EasyDict

configer = EasyDict()

configer.ckptdir = './ckpt'
configer.logdir = './log'

configer.inputsize = (3, 112, 96)    # (C, H, W)
configer.batchsize = 96
configer.n_epoch = 40

configer.lrbase = 0.0004
configer.adjstep = [30,]
configer.gamma = 0.1

configer.cuda = True

