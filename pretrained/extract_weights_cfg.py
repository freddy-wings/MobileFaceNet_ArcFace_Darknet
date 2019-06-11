import os
import numpy as np

import torch
from torch import nn
from tensorboardX import SummaryWriter

from mobilefacenet import MobileFacenet, ConvBlock, Bottleneck


def _read_ckpt(prefix):

    net = MobileFacenet(num_classes=10)
    state = torch.load(prefix, map_location='cpu')
    net_state = state['net']
    net_state = {k: v for k, v in net_state.items() if k!='weight'}
    toload = net.state_dict(); toload.update(net_state)
    net.load_state_dict(toload)
    
    # logdir = './pretrained/'
    # for f in os.listdir(logdir):
    #     if 'events' in f:
    #         return net

    # with SummaryWriter(log_dir=logdir, comment='mobilefacenet') as w:
    #     dummy_input = torch.rand(2, 3, 112, 96)
    #     w.add_graph(net, (dummy_input, ))

    return net



def _extract_conv(fp, module):
    if module.bias is not None:
        module.bias.data.numpy().tofile(fp)
    else:
        bias = torch.zeros(module.weight.shape[0]).float()
        bias.data.numpy().tofile(fp)
    module.weight.data.numpy().tofile(fp)

def _extract_bn(fp, module):
    module.bias.data.numpy().tofile(fp)
    module.weight.data.numpy().tofile(fp)
    module.running_mean.numpy().tofile(fp)
    module.running_var.numpy().tofile(fp)

def _extract_prelu(fp, module):
    module.weight.data.numpy().tofile(fp)

def _extract_locally_connected(fp, module):
    """
    Notes:
        locally connected layer:
            - input:    (batch, inputs)
            - output:   (batch, outputs)
            - weights:  (groups, outputs/groups, inputs/groups)
            - biases:   (outputs)
    """
    if module.bias is not None:
        module.bias.data.numpy().tofile(fp)
    else:
        bias = torch.zeros(module.weight.shape[0]).float()
        bias.data.numpy().tofile(fp)
    
    groups = module.groups
    inputs = np.prod(module.kernel_size) * groups
    outputs = 1 * groups

    weights = module.weight.data.numpy()
    weights = weights.reshape((groups, outputs//groups, inputs//groups))
    weights.tofile(fp)


def _extract_ConvBlock(fp, module, global_conv=False):

    if global_conv:
        _extract_locally_connected(fp, module.conv)
    else:
        _extract_conv(fp, module.conv)
    _extract_bn(fp, module.bn)
    if not module.linear:
        _extract_prelu(fp, module.prelu)

def _extract_Bottleneck(fp, module):

    for submodule in module.conv.children():
        mt = type(submodule)
        if mt == nn.Conv2d:
            _extract_conv(fp, submodule)
        elif mt == nn.BatchNorm2d:
            _extract_bn(fp, submodule)
        elif mt == nn.PReLU:
            _extract_prelu(fp, submodule)



def extract_mobilefacenet_weights(ckpt):

    net = _read_ckpt(ckpt)
    fp = open('../weights/mobilefacenet.weights', 'wb')

    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)

    for name, module in net.named_children():
        print(name)
        mt = type(module)
        if mt == ConvBlock:
            _extract_ConvBlock(fp, module, global_conv=True if name == 'linear7' else False)
        elif mt == nn.Sequential:
            for submodule in module:
                _extract_Bottleneck(fp, submodule)

    fp.close()








def _write_conv_cfg(fp, module):

    filters = module.out_channels
    size = module.kernel_size[0]
    stride = module.stride[0]
    padding = module.padding[0]
    groups = module.groups

    cfg = []
    cfg += ['[convolutional]']
    cfg += ['filters=%d' % filters]
    cfg += ['size=%d' % size]
    cfg += ['stride=%d' % stride]
    cfg += ['padding=%d' % padding]
    cfg += ['groups=%d' % groups]
    cfg += ['activation=linear']
    cfg += ['']

    cfg = map(lambda x: x + '\n', cfg)
    fp.writelines(cfg)

def _write_bn_cfg(fp, module):
    
    cfg = []
    cfg += ['[batchnorm]']
    cfg += ['bias=1']
    cfg += ['']

    cfg = map(lambda x: x + '\n', cfg)
    fp.writelines(cfg)

def _write_prelu_cfg(fp, module):
    
    n = module.num_parameters

    cfg = []
    cfg += ['[prelu]']
    cfg += ['n=%d' % n]
    cfg += ['']

    cfg = map(lambda x: x + '\n', cfg)
    fp.writelines(cfg)

def _write_locally_connected_cfg(fp, module):
    
    output = module.weight.shape[0]
    groups = module.groups
    activation = 'linear'

    cfg = []
    cfg += ['[connected-locally]']
    cfg += ['output=%d' % output]
    cfg += ['groups=%d' % groups]
    cfg += ['activation=%s' % activation]
    cfg += ['']

    cfg = map(lambda x: x + '\n', cfg)
    fp.writelines(cfg)



def _write_net_cfg(fp, module):

    cfg = []
    cfg += ['[net]']
    cfg += ['height=112']
    cfg += ['width=96']
    cfg += ['channels=3']
    cfg += ['']

    cfg = map(lambda x: x + '\n', cfg)
    fp.writelines(cfg)

def _write_ConvBlock_cfg(fp, module, global_conv=False):

    if global_conv:
        _write_locally_connected_cfg(fp, module.conv)
        # _write_conv_cfg(fp, module.conv)
    else:
        _write_conv_cfg(fp, module.conv)
    _write_bn_cfg(fp, module.bn)
    if not module.linear:
        _write_prelu_cfg(fp, module.prelu)

def _write_Bottleneck_cfg(fp, module, index):

    connect = module.connect
    for submodule in module.conv.children():
        mt = type(submodule)

        if mt == nn.Conv2d:
            _write_conv_cfg(fp, submodule)
        
        elif mt == nn.BatchNorm2d:
            _write_bn_cfg(fp, submodule)
        
        elif mt == nn.PReLU:
            _write_prelu_cfg(fp, submodule)
    

def write_mobilefacenet_cfg():

    net = _read_ckpt('./pretrained/MobileFacenet_best.pkl')
    fp = open('../cfg/mobilefacenet.cfg', 'w')
    index = 0

    _write_net_cfg(fp, net)
    tmp = list(net.children())
    for name, module in net.named_children():
        print(module)
        mt = type(module)

        if mt == ConvBlock:
            fp.write('\n# layer %d: ConvBlock\n' % index)
            _write_ConvBlock_cfg(fp, module, global_conv=True if name == 'linear7' else False)
            index += 3
            
        elif mt == nn.Sequential:
            for submodule in module:
                fp.write('\n# layer %d: Bottleneck\n' % index)
                _write_Bottleneck_cfg(fp, submodule, index)

                if submodule.connect:
                    cfg = []
                    cfg += ['[shortcut]']
                    cfg += ['from=%d' % (index-1)]
                    cfg += ['activation=linear']

                    cfg = map(lambda x: x + '\n', cfg)
                    fp.writelines(cfg)
                    index += 1
                    
                index += 8

    fp.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="extract weights & generate cfg file")
    parser.add_argument('--pretrained', '-p', required=True, help='pretrained model path')
    parser.add_argument('--cfgfile', '-cfg', action='store_true', help='generate .cfg file if set True')

    args = parser.parse_args()
    extract_mobilefacenet_weights(args.pretrained)
    if args.cfgfile:
        write_mobilefacenet_cfg()
    