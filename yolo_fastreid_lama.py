import cv2
import os
import pandas as pd
import hydra
import numpy as np
import torch.cuda
from PIL import Image
import sys
sys.path.append(".")
import math
from yolov7 import yolo_fastreid
time_interval=1   #转视频帧间隔

def colorTowhite(file_pathname,output):
    #遍历该目录下的所有图片文件
    kernel_2 = np.ones((30, 30), dtype=np.uint8)  # 卷积核变为4*4

    for filename in os.listdir(file_pathname):
        mask = cv2.imread(file_pathname+'/'+filename,0)
        mask_2=np.copy(mask)
        mask_2[mask>=10]=255
        dilate = cv2.dilate(mask_2, kernel_2, 15)
        cv2.imwrite(output+"/"+filename[0:-4]+"_mask"+".png",dilate)
    print("已完成mask转换")

# 图片转视频
def frame2video(video_dir, fps):
    im_dir=".../example/lama_s_pic/"
    im_list = os.listdir(im_dir)
    im_list = sorted(im_list)
    #im_list.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  #最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('已完成图片转视频')
#视频转图片
def video2frame(videos_path,time_interval):
    '''
  :param videos_path: 视频的存放路径
  :param frames_save_path: 视频切分成帧之后图片的保存路径
  :param time_interval: 保存间隔
  :return:
'''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            cv2.imencode('.jpg', image)[1].tofile("lama/test-image" + "/frame%d.png" % count)
        # if count == :
        #     break
    print('完成视频转图片')
def zhuanhuan(input_pathname,output_name,a,b):
    #转换文件目录
    for filename in os.listdir(input_pathname)[a:b]:
        mask = cv2.imread(input_pathname+'/'+filename,1)
        cv2.imwrite(output_name+"/"+filename,mask)
    print("已完成文件夹转换")
def removepicture(pathname):
    for pic in os.listdir(pathname):
        if pic.endswith(".png"):
            os.remove(pathname + '/' + pic)
        elif pic.endswith(".jpg"):
            os.remove(pathname + '/' + pic)
        elif pic.endswith(".txt"):
            os.remove(pathname + '/' + pic)
    print("删除完成")
def read_txt(file_pathname,filename):
    data = pd.read_csv(file_pathname+'/'+"labels"+'/'+filename[0:-4]+".txt", sep=' ', names=["num", "x1", "y1", "x2", "y2"])
    x1=data.x1[0]#左上
    y1=data.y1[0]
    x2 = data.x2[0]#右下
    y2 = data.y2[0]
    return x1,y1,x2,y2

def make_mask(file_pathname,output):
    for filename in os.listdir(file_pathname):
        if (filename.endswith(".png") or filename.endswith(".jpg")):
            yuantu= cv2.imread(file_pathname+'/'+filename,1)
            img_size=yuantu.shape
            h=img_size[0]
            w=img_size[1]
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2=read_txt(file_pathname,filename)
        #补一个边界判断
            mask[y1-10:y2+10, x1-10:x2+10] = 255
            cv2.imwrite(output + "/" + filename[0:-4] + "_mask" + ".png", mask)
###################################################################################
def yolo_fastreid_lama(chushi_path):

    # 将视频转换为帧
    step=5
    num_pic = len([f for f in os.listdir(".../example/input-image/" + chushi_path) if f.endswith(".jpg") or f.endswith(".jpg")])
    removepicture(".../lama/test-image")
    removepicture(".../lama/out-put")
    removepicture(".../yolov7/inference/myself-detect/yolo-reid")
    removepicture(".../runs/detect/exp/labels")
    zhuanhuan(".../example/input-image/" + chushi_path, ".../yolov7/inference/myself-detect/yolo-reid", 0, num_pic)

    yolo_fastreid.detect()

if __name__ == '__main__':
    print(torch.cuda.current_device())
    fps = 30  # 视频帧数/秒
    chushi_path="reid-3"
    yolo_fastreid_lama(chushi_path)
    frame2video("example/output-vedio/" + chushi_path + '_steg_3'+".mp4", fps)








