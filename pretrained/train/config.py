from easydict import EasyDict

configer = EasyDict()

configer.ckptdir = './ckpt'
configer.logdir = './log'

configer.inputsize = (3, 112, 96)    # (C, H, W)
configer.batchsize = 256
configer.n_epoch = 70

configer.lrbase = 0.01
configer.adjstep = [36, 52, 58]
configer.gamma = 0.1

configer.cuda = True

