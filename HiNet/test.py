import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义加载模型的函数
def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

# 定义生成高斯噪声的函数
def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

#定义计算 PSNR（峰值信噪比）的函数
def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)  # 使用 DataParallel 并行化模型

params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))# 获取可训练参数
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)  # 定义 Adam 优化器
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma) # 定义学习率调度器

load(c.MODEL_PATH + c.suffix)

net.eval()

dwt = common.DWT()
iwt = common.IWT()

# 在评估模式下不计算梯度
with torch.no_grad():
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        cover = data[data.shape[0] // 2:, :, :, :]#cover 是数据集的后半部分，secret 是数据集的前半部分。
        secret = data[:data.shape[0] // 2, :, :, :]
        cover_input = dwt(cover)#使用离散小波变换（DWT）将 cover 和 secret 转换为频域表示，分
        # 别得到 cover_input 和 secret_input。
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = iwt(output_steg)
        backward_z = gauss_noise(output_z.shape)#生成高斯噪声
#将 cover_input 和 secret_input 拼接成 input_img，作为神经网络的输入。
#通过神经网络前向传播，得到 output。
#output_steg 是 output 的前四个通道，表示包含秘密信息的图像。
#使用逆离散小波变换（IWT）将 output_steg 转换回空间域，得到 steg_img。
        #################
        #   backward:   #
        #################
        output_rev = torch.cat((output_steg, backward_z), 1) # 拼接嵌入图像和高斯噪声
        bacward_img = net(output_rev, rev=True)# 通过网络反向传播
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)# 获取还原的秘密图像
        secret_rev = iwt(secret_rev)# 对还原的秘密图像进行逆 DWT 变换
        cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in) # 获取还原的封面图像
        cover_rev = iwt(cover_rev)# 对还原的封面图像进行逆 DWT 变换
        resi_cover = (steg_img - cover) * 20 # 计算封面图像的残差
        resi_secret = (secret_rev - secret) * 20 # 计算秘密图像的残差
#保存图像
        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)




