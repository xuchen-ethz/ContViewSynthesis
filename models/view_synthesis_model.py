import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import itertools
import projection_layer
from networks import rotation_tensor

class ViewSynthesisModel(BaseModel):
    def name(self):
        return 'ViewSynthesisModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # load/define networks

        self.netG = networks.FTAE(opt.input_nc, opt.ngf, n_layers=int(np.log2(opt.fineSize)), n_bilinear_layers=opt.n_bilinear_layers,
                         norm_layer=networks.get_norm_layer(norm_type=opt.norm),
                         nl_layer_enc=networks.get_non_linearity(layer_type=opt.nl_enc),
                         nl_layer_dec=networks.get_non_linearity(layer_type=opt.nl_dec),gpu_ids=opt.gpu_ids,nz=opt.nz)

        self.netR = networks.AE(opt.input_nc*2, ndf=64, n_layers=int(np.log2(opt.fineSize)), norm_layer=networks.get_norm_layer(norm_type=opt.norm),
                         nl_layer_enc=networks.get_non_linearity(layer_type=opt.nl_enc), gpu_ids=opt.gpu_ids)
        if len(opt.gpu_ids) > 0:
            self.netG.cuda(opt.gpu_ids[0])
            self.netR.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netG, init_type="normal")
        networks.init_weights(self.netR, init_type="normal")


        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netR, 'R', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(),self.netR.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        zeros = Variable(torch.zeros((opt.batchSize,1)).cuda())
        ones = Variable(torch.ones((opt.batchSize,1)).cuda())

        cam_param = np.loadtxt(os.path.join(self.opt.dataroot,'cam_param.txt'))

        self.pose_abs = torch.cat([cam_param[0] * ones, cam_param[1] * ones, cam_param[2] * ones,
                                   cam_param[3] * ones, cam_param[4] * ones, cam_param[5] * ones], dim=1).cuda()
        self.dist = cam_param[2] / np.cos(cam_param[3])
        sensor_size = cam_param[6]
        focal_length = cam_param[7]

        intrinsics = np.array(
            [opt.fineSize / sensor_size * focal_length, 0., opt.fineSize / 2., \
             0., opt.fineSize / sensor_size * focal_length, opt.fineSize / 2., \
             0., 0., 1.]).reshape((3, 3))
        intrinsics_inv = np.linalg.inv(intrinsics)

        self.intrinsics = Variable(torch.from_numpy(intrinsics.astype(np.float32)).cuda()).unsqueeze(0).expand(opt.batchSize,3,3)
        self.intrinsics_inv = Variable(torch.from_numpy(intrinsics_inv.astype(np.float32)).cuda()).unsqueeze(0).expand(opt.batchSize,3,3)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netR)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

        if self.opt.isTrain:
            input_B = input['B']
            input_C = input['C']
            input_YawAB = input['YawAB']
            input_YawCB = input['YawCB']

            if len(self.gpu_ids) > 0:
                input_B = input_B.cuda(self.gpu_ids[0], async=True)
                input_C = input_C.cuda(self.gpu_ids[0], async=True)

                input_YawAB = input_YawAB.cuda(self.gpu_ids[0], async=True)
                input_YawCB = input_YawCB.cuda(self.gpu_ids[0], async=True)

            self.input_B = input_B
            self.input_C = input_C

            self.input_YawAB = input_YawAB
            self.input_YawCB = input_YawCB


    def forward(self):

        self.real_A = Variable(self.input_A,volatile=True)
        self.real_B = Variable(self.input_B,volatile=True)
        self.real_C = Variable(self.input_C,volatile=True)

        self.real_YawAB= Variable(self.input_YawAB,volatile=True)
        self.real_YawCB = Variable(self.input_YawCB,volatile=True)

        self.pred_YawAB = self.netR( torch.cat([self.real_A, self.real_B],dim=1) ) # [rebuttal]

        self.loss_R = self.criterionL1(self.pred_YawAB,self.real_YawAB)

        b,c,h,w = self.real_A.size()
        zeros = Variable(torch.zeros((b,1)).cuda() )

        pose_rel = torch.cat( [zeros,zeros,zeros,zeros,-self.pred_YawAB, zeros], dim=1)

        R = rotation_tensor(zeros, zeros, self.pred_YawAB).cuda()

        self.depth = self.netG(self.real_A, R)
        self.depth = self.depth + self.dist

        self.fake_B_flow = projection_layer.inverse_warp(self.real_C, self.depth,
                                                                   pose_rel, self.pose_abs[:b,:], self.intrinsics[:b,:,:], self.intrinsics_inv[:b,:,:])

        self.fake_B = F.grid_sample(self.real_C, self.fake_B_flow)

    def test(self):

        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B_list = []

            NV = self.opt.test_views
            b,c,h,w = self.real_A.size()
            zeros = Variable(torch.zeros((b,1)).cuda(),volatile=True )

            whole_range = 3*np.pi/9.  # 80
            yaw = -3*np.pi/18. - whole_range / NV
            for i in range(NV):
                yaw += whole_range/NV

                self.real_Yaw = Variable(-torch.Tensor([yaw]).cuda(self.gpu_ids[0], async=True) ,volatile=True).unsqueeze(0)
                pose_rel = torch.cat([zeros, zeros, zeros, zeros, -self.real_Yaw, zeros], dim=1)

                R = rotation_tensor(zeros, zeros, self.real_Yaw).cuda()

                self.depth = self.netG(self.real_A, R) + self.dist
                self.fake_B_flow = projection_layer.inverse_warp(self.real_A, self.depth,
                                                                           pose_rel, self.pose_abs[:b, :],
                                                                            self.intrinsics[:b, :, :],
                                                                            self.intrinsics_inv[:b, :, :])

                self.fake_B = F.grid_sample(self.real_A, self.fake_B_flow)

                self.fake_B_list.append(self.fake_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        self.loss_TV = self.criterionTV(self.depth) * self.opt.lambda_tv

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_recon

        self.loss_G = self.loss_G_L1 + self.loss_TV

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_L1', self.loss_G_L1.data[0]),
                            ('TV', self.loss_TV.data[0]),
                            ('R', self.loss_R.data[0]),

                            ])

    def get_current_visuals(self):
        if not self.opt.isTrain:
            return self.get_current_visuals_test()
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_C = util.tensor2im(self.real_C.data)

        flow = util.tensor2im(self.fake_B_flow.permute(0,3,1,2).data)
        depth = util.tensor2im_depth(self.depth.data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),('real_C', real_C),
                        ('flow',flow), ('depth', depth)])

    def get_current_visuals_test(self):
        visual_list = OrderedDict([])
        for idx,fake_B_var in zip(range(len(self.fake_B_list)), self.fake_B_list):
            visual_list['%03d'%idx] = util.tensor2im(fake_B_var.data)
        return visual_list

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
