import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import HiNet.config as c
import datasets
import HiNet.modules.Unet_common as common


class Steganography:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Model().to(self.device)
        init_model(self.net)
        self.net = torch.nn.DataParallel(self.net, device_ids=c.device_ids)

        params_trainable = list(filter(lambda p: p.requires_grad, self.net.parameters()))
        self.optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, c.weight_step, gamma=c.gamma)

        self.dwt = common.DWT()
        self.iwt = common.IWT()

        self.load_model(c.MODEL_PATH + c.suffix)

    def load_model(self, name):
        state_dicts = torch.load(name)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.net.load_state_dict(network_state_dict)
        try:
            self.optim.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')

    def gauss_noise(self, shape):
        noise = torch.zeros(shape).to(self.device)
        for i in range(noise.shape[0]):
            noise[i] = torch.randn(noise[i].shape).to(self.device)
        return noise

    def computePSNR(self, origin, pred):
        origin = np.array(origin).astype(np.float32)
        pred = np.array(pred).astype(np.float32)
        mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
        if mse < 1.0e-10:
            return 100
        return 10 * math.log10(255.0 ** 2 / mse)

    @torch.no_grad()
    def steganography(self, cover, secret):
        cover_input = self.dwt(cover)
        secret_input = self.dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        # 前向传播
        output = self.net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        steg_img = self.iwt(output_steg)

        return steg_img

    @torch.no_grad()
    def recovery(self, steg_img):
        cover_input = self.dwt(steg_img)
        backward_z = self.gauss_noise(cover_input.shape)
        output_rev = torch.cat((cover_input, backward_z), 1)
        backward_img = self.net(output_rev, rev=True)
        secret_rev = backward_img.narrow(1, 4 * c.channels_in, backward_img.shape[1] - 4 * c.channels_in)
        secret_rev = self.iwt(secret_rev)
        cover_rev = backward_img.narrow(1, 0, 4 * c.channels_in)
        cover_rev = self.iwt(cover_rev)

        return cover_rev, secret_rev