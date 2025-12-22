import os
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision
import itertools
from tqdm import tqdm


idx = 99999

def visualize(img_arr, dpi=80):
    plt.figure(figsize=(10,10), dpi=dpi)
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    plt.show()

def load_image(filename):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])

    img = Image.open(filename)
    img = transform(img)
    return img.unsqueeze(dim=0)

# 定义了对抗损失的计算方式
class GANLoss(nn.Module):
    def __init__(self, LSGAN=True):
        super(GANLoss, self).__init__()
        # 两个label定义了真标签(1)和假标签(0)
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        if LSGAN:
            # LSGAN损失
            self.loss = nn.MSELoss()
        else:
            # 标准的对抗损失
            self.loss = nn.BCEWithLogitsLoss()


    def forward(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        # 将label的维度扩展为与输入的特征一致，例如 30*30
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0)]
        conv_block += [nn.InstanceNorm2d(dim), nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0)]
        conv_block += [nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = x + self.conv_block(x)
        return self.relu(out)

class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        # 编码器
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        # 转换器
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # 解码器
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        sequence = [
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

datasetA = dset.ImageFolder(root='./data/dataA/',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloaderA = DataLoader(datasetA, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

datasetB = dset.ImageFolder(root='./data/dataB/',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloaderB = DataLoader(datasetB, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

iteratorA = iter(dataloaderA)
dataA, label = next(iteratorA)
iteratorB = iter(dataloaderB)
dataB, label = next(iteratorB)

class CycleGANModel(nn.Module):
    def __init__(self):
        super(CycleGANModel, self).__init__()

        self.netG_A = ResnetGenerator(ngf=64, n_blocks=9)
        self.netG_B = ResnetGenerator(ngf=64, n_blocks=9)
        self.netD_A = NLayerDiscriminator(n_layers=3, ndf=64)
        self.netD_B = NLayerDiscriminator(n_layers=3, ndf=64)


        self.criterionGAN = GANLoss()
        self.criterionCycle = torch.nn.L1Loss()
        # itertools.chain 就多组参数视为整体一起优化
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.0002, betas=(0.5, 0.999))


    def forward(self, real_A, real_B):
        fake_B = self.netG_A(real_A)   # G_A(A)
        rec_A = self.netG_B(fake_B)    # G_B(G_A(A))
        fake_A = self.netG_B(real_B)   # G_B(B)
        rec_B = self.netG_A(fake_A)    # G_A(G_B(B))
        return [fake_B, rec_A, fake_A, rec_B]

    # 计算判别器D的损失并更新判别器
    def backward_D(self, netD, real, fake):
        # 对于生成图像，D的目标是将其判断为假
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # 对于真实图像，D的目标是将其判断为真
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D.data.mean().item()

    # 计算生成器G的损失并更新生成器
    def backward_G(self, real_A, real_B, fake_B, rec_A, fake_A, rec_B):
        # G的目标是让D网络判断生成图像为真
        loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
        # 前向的循环一致性损失 || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * 10.0
        # 反向的循环一致性损失 || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * 10.0
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        return loss_G_A.data.mean().item(), loss_G_B.data.mean().item(), loss_cycle_A.data.mean().item(), loss_cycle_B.data.mean().item()

    # 训练流程
    def optimize_parameters(self, real_A, real_B):
        [fake_B, rec_A, fake_A, rec_B] = self.forward(real_A, real_B)

        self.optimizer_G.zero_grad()
        loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B = self.backward_G(real_A, real_B, fake_B, rec_A, fake_A, rec_B)
        self.optimizer_G.step()
        # 更新D网络
        self.optimizer_D.zero_grad()
        loss_D_A = self.backward_D(self.netD_A, real_B, fake_B.detach())
        loss_D_B = self.backward_D(self.netD_B, real_A, fake_A.detach())
        self.optimizer_D.step()
        return fake_B, rec_A, fake_A, rec_B, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_D_A, loss_D_B


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CycleGANModel().to(device)
model.eval()
model.load_state_dict(torch.load(f"checkpoint/cyclegan-%05d.pt"%(idx)))

As = ['harvard_0.jpg', 'harvard_1.jpg', 'harvard_2.jpg', 'harvard_3.jpg']
Bs = ['landscape_001.jpg', 'landscape_002.jpg', 'landscape_003.jpg', 'landscape_004.jpg']

model.eval()
for i in range(4):
    I_A = load_image('./data/dataA/trainA/' + As[i])
    I_B = load_image('./data/dataB/trainB/' + Bs[i])
    with torch.no_grad():
        fake_B, rec_A, fake_A, rec_B = model(I_A.cuda(), I_B.cuda())
    visualize(torch.cat((I_A[0], fake_B[0].detach().cpu(), rec_A[0].detach().cpu(),
                         I_B[0], fake_A[0].detach().cpu(), rec_B[0].detach().cpu()), dim=2).cpu(), 120)