import cv2
import os
import pandas as pd
import hydra
import numpy as np
import torch.cuda
from PIL import Image
import sys
sys.path.append(".")

from HiNet.steganography import Steganography
from HiNet import config as con
time_interval=1   #转视频帧间隔
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
def frame2video(im_dir,video_dir, fps):

    im_list = os.listdir(im_dir+"/")
    im_list = sorted(im_list)
    #im_list.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  #最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir+'/' + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('已完成图片转视频')
def read_txt(file_pathname,filename):
    data = pd.read_csv(file_pathname+filename+".txt", sep=' ', names=["x1", "y1", "x2", "y2"])
    x1=data.x1[0]#左上
    y1=data.y1[0]
    x2 = data.x2[0]#右下
    y2 = data.y2[0]
    return x1,y1,x2,y2

if __name__ == '__main__':
    preprocess = T.Compose([
        T.CenterCrop(con.cropsize_val),
        T.ToTensor(),
    ])

    fps = 30  # 视频帧数/秒
    chushi_path="reid-3"
    im_dir = ".../example/lama_s_pic"
    recovered_cover_path=".../example/cover-rec"
    recovered_secret_path=".../example/secret-rec"
    cover_s_r_path=".../example/cover_sec_rec"#保存最终图片的地址

    # 初始化 Steganography 类
    steg = Steganography()
    steg.net.eval()
    cover_list=os.listdir(im_dir)
    cover_list = sorted(cover_list)#封面图像
    num_rec=1
    for filename in os.listdir(im_dir):
        steg_image_pil = Image.open(im_dir+'/'+filename)
        steg_image = preprocess(steg_image_pil).unsqueeze(0).cuda()
        recovered_cover_image, recovered_secret_image = steg.recovery(steg_image)
        torchvision.utils.save_image(recovered_cover_image, recovered_cover_path+'/'+filename[0:-12]+"_rec.png")  # 恢复后的封面图像
        torchvision.utils.save_image(recovered_secret_image,recovered_secret_path+'/'+filename[0:-12]+"_rec.png")  # 恢复后的秘密图像
        print(f'恢复图像,第{num_rec}张')
        num_rec+=1
    secret_list=os.listdir(recovered_secret_path)
    secret_list = sorted(secret_list)#恢复的秘密图像
    for i in range(len(secret_list)):
        filename=secret_list[i]
        filename_cov=cover_list[i]
        xmin,ymin,xmax ,ymax=read_txt(".../runs/detect/exp/labels/",filename[0:-8])
        w = xmax - xmin
        h = ymax - ymin
        transf = T.Compose([
            T.CenterCrop((h,w))
        ])
        rec_pil = Image.open(recovered_secret_path + '/' + filename)
        rec_pil = transf(rec_pil)
        cov_pil=Image.open(im_dir + '/' + filename_cov)
        box=(xmin,ymin,xmax ,ymax)
        cov_pil.paste(rec_pil, box)#将秘密图像粘贴到封面图像指定位置
        cov_pil.save(cover_s_r_path+'/'+filename[0:-8]+'_final.png')
    frame2video(cover_s_r_path,".../example/output-vedio"+'/'+"reid-36"+"_final.mp4",25)


