import torch
import torchvision
import config as c
import datasets
from steganography import Steganography
import torchvision.transforms as T
from PIL import Image



# 初始化 Steganography 类
steg = Steganography()
steg.net.eval()
#读取图片并处理图片
preprocess = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])
cover_image_path = "val/cover/frame2612.jpg"
secret_image_path = "val/secrte/0001_c1s1_0_04.jpg"
cover_image = preprocess(Image.open(cover_image_path)).unsqueeze(0).cuda()  # 加载并转为批量形式
secret_image = preprocess(Image.open(secret_image_path)).unsqueeze(0).cuda()  # 加载并转为批量形式
#隐写
steg_image = steg.steganography(cover_image, secret_image)
# 执行恢复
recovered_cover_image, recovered_secret_image = steg.recovery(steg_image)
# 将图像保存为文件（可选）
torchvision.utils.save_image(steg_image, c.IMAGE_PATH_steg+"steg_image2.png")#含有秘密的封面图像
torchvision.utils.save_image(recovered_cover_image, "image/cover-rev/recovered_cover_image2.png")#恢复后的封面图像
torchvision.utils.save_image(recovered_secret_image, c.IMAGE_PATH_secret_rev+"recovered_secret_image2.png")#恢复后的秘密图像