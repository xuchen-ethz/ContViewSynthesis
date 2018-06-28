import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset,make_dataset_label
from PIL import Image
import numpy as np

class AlignedDatasetMultiView(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dirs = []
        self.paths = []

        if self.opt.isTrain:
            self.nv = 0
            for d in os.listdir(self.root):
                if os.path.isdir(os.path.join(self.root,d)):
                    self.nv += 1

            for i in range(self.nv):
                self.dirs.append(os.path.join(opt.dataroot, "%d" %i) )
                self.paths.append(sorted(make_dataset(self.dirs[i]) ) )
        else:
            self.dir = (os.path.join(opt.dataroot))
            self.paths = (sorted(make_dataset(self.dir)))

        self.transform = get_transform(opt)

    def __getitem__(self, index):

        if self.opt.phase == 'test' :
            bg_color = (64, 64, 64)
            A = Image.open(self.paths[index]).convert('RGB')
            A, _ = self.remapping_background(A, bg_color)
            A = self.transform(A)
            return {'A': A, 'A_paths': self.paths[index], }


        training_view_indexes = range(0,self.nv)

        idx_A = np.random.choice(training_view_indexes)
        idx_B = np.random.choice(training_view_indexes)
        delta_choices = [2,1,-1,-2]
        idx_C = idx_B + np.random.choice(delta_choices)

        # print idx_A, idx_B, idx_C
        yaw1 = -(idx_B-idx_A) * np.pi/9
        yaw2 = -(idx_B-idx_C) * np.pi/9
        idx_C = np.mod(idx_C,self.nv)



        bg_color = (64,64,64)
        A = Image.open(self.paths[idx_A][index]).convert('RGB')
        A,_ = self.remapping_background(A, bg_color)
        A = self.transform(A)

        B = Image.open(self.paths[idx_B][index]).convert('RGB')
        B,_ = self.remapping_background(B, bg_color)
        B = self.transform(B)

        C = Image.open(self.paths[idx_C][index]).convert('RGB')
        C,_ = self.remapping_background(C, bg_color)
        C = self.transform(C)

        return {'A': A, 'B': B, 'C': C, 'YawAB': torch.Tensor([yaw1]),'YawCB': torch.Tensor([yaw2]), 'A_paths': self.paths[int(self.nv/2)][index], }

    def __len__(self):
        if self.opt.phase == 'train':
            return len(self.paths[int(self.nv/2)])
        else:
            return len(self.paths)

    def name(self):
        return 'AlignedDatasetMultiView'

    def remapping_background(self, image, bg_color):
        data = np.array(image)

        r1, g1, b1 = bg_color  # Original value
        r2, g2, b2 = 128, 128, 128  # Value that we want to replace it with

        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:, :, :3][mask] = [r2, g2, b2]

        return Image.fromarray(data),mask

