import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F # for cross entropy loss
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple
###############################################################################
# Functions
###############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def maxpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

def convMaxpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

# Defines the total variation loss
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, input):

        w_variance = torch.mean((input[:, :, :, 1:] - input[:, :, :, :-1]) ** 2)
        h_variance = torch.mean((input[:, :, 1:, :] - input[:, :, :-1, :]) ** 2)
        loss = (w_variance + h_variance)
        return loss

def rotation_z_tensor(yaw, n_comps, gpu):
    yaw = yaw.unsqueeze(1)
    one = Variable(torch.ones(n_comps, 1, 1).cuda(gpu, async=True))
    zero = Variable(torch.zeros(n_comps, 1, 1).cuda(gpu, async=True))

    # print yaw, one, zero
    rot_z = torch.cat((
        torch.cat((yaw.cos(), -yaw.sin(), zero), 1),
        torch.cat((yaw.sin(), yaw.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return rot_z

def rotation_tensor(theta, phi, psi):
    n_comps = theta.size(0)
    theta = theta.unsqueeze(1)
    phi = phi.unsqueeze(1)
    psi = psi.unsqueeze(1)

    one = Variable(torch.ones(n_comps, 1, 1)).cuda()
    zero = Variable(torch.zeros(n_comps, 1, 1)).cuda()
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))

def rotation_x_tensor(yaw, n_comps, gpu):
    yaw = yaw.unsqueeze(1)
    one = Variable(torch.ones(n_comps, 1, 1).cuda(gpu, async=True))
    zero = Variable(torch.zeros(n_comps, 1, 1).cuda(gpu, async=True))

    # print yaw, one, zero
    rot_z = torch.cat((
        torch.cat((yaw.cos(), zero, -yaw.sin()), 1),
        torch.cat((yaw.sin(), yaw.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return rot_z

class FTAE(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7, n_bilinear_layers=0,
                 norm_layer=None, nl_layer_enc=None, nl_layer_dec=None, gpu_ids=[], nz=200):
        super(FTAE, self).__init__()
        self.gpu_ids = gpu_ids
        kw, padw = 4, 1

        enc = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer_enc()]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            enc += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and n < n_layers - 1:
                enc += [norm_layer(ndf * nf_mult)]
                enc += [nl_layer_enc()]
        # sequence += [nn.AvgPool2d(8)]
        self.enc = nn.Sequential(*enc)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, nz * 3)]) #, nn.LeakyReLU(0.2, True),
        self.fc2 = nn.Sequential(*[nn.Linear(nz * 3, ndf * nf_mult),nl_layer_enc()]) #, nn.BatchNorm1d(ndf * nf_mult)

        deconv = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 4)

            upsample = 'bilinear' if n_layers - n < n_bilinear_layers else 'basic'
            deconv += upsampleLayer(ndf * nf_mult_prev, ndf * nf_mult, upsample=upsample)
            if norm_layer is not None and (n_layers - n + 1) < n_layers:
                deconv += [norm_layer(ndf * nf_mult)]
            deconv += [nl_layer_dec()]

        output_nc = 1
        if n_bilinear_layers > 0:
            deconv += upsampleLayer(ndf, output_nc, upsample='bilinear')
        else:
            deconv += upsampleLayer(ndf, output_nc, upsample='basic')

        self.deconv = nn.Sequential(*deconv)
        self.nz = nz

    def forward(self, x, R):
        z_conv = self.enc(x)

        z_fc = self.fc(z_conv.view(x.size(0),-1) ).view(x.size(0), self.nz, 3)
        z_fc = F.tanh(z_fc)

        z_rot = z_fc.bmm(R)
        z_rot_fc = self.fc2(z_rot.view(x.size(0), self.nz*3))

        output = self.deconv(z_rot_fc.view(z_conv.size(0),z_conv.size(1),z_conv.size(2),z_conv.size(3)))

        return output


class AE(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7,
                 norm_layer=None, nl_layer_enc=None, gpu_ids=[]):
        super(FTAE, self).__init__()
        self.gpu_ids = gpu_ids
        kw, padw = 4, 1

        enc = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer_enc()]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            enc += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and n < n_layers - 1:
                enc += [norm_layer(ndf * nf_mult)]
                enc += [nl_layer_enc()]

        enc += [nn.Linear(ndf * nf_mult, 1)]

        self.enc = nn.Sequential(*enc)

    def forward(self, x, R):
        output = self.enc(x)
        output = output / (2*np.pi)
        output = F.tanh(output)
        return output

