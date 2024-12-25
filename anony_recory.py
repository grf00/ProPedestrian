import cv2
import os
import pandas as pd
import hydra
import numpy as np
import torch.cuda
from PIL import Image
import sys
sys.path.append(".")
sys.path.append( "/home/lhb/grf" )
sys.path.append( "/home/lhb/grf/lama" )
sys.path.append( "lama/configs" )
sys.path.append( "lama/models" )
sys.path.append( "/home/lhb/grf/HiNet")
sys.path.append( "/home/lhb/grf/yolov7" )
sys.path.append( "yolov7/fast_reid_master" )
time_interval=1   #转视频帧间隔
import cv2
import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from HiNet.steganography import Steganography
from HiNet import config as con

class recover_images:
    def __init__(self,fps=30,im_dir="example/lama_s_pic"):
        self.im_dir = im_dir
        self.recovered_cover_path = "example/cover-rec"
        self.recovered_secret_path = "example/secret-rec"
        self.cover_s_r_path = "example/cover_sec_rec"
        self.label_path = "runs/detect/exp/labels"
        self.output_video_path = "example/output-vedio/reid-street_final.mp4"
        self.fps = fps
        self.steg = Steganography()
        self.steg.net.eval()

    @staticmethod
    def remove_files(path):
        for file in os.listdir(path):
            if file.endswith(('.png', '.jpg', '.txt')):
                os.remove(os.path.join(path, file))
        print(f"Files removed from {path}")

    @staticmethod
    def read_txt(file_path):
        data = pd.read_csv(file_path, sep=' ', names=["x1", "y1", "x2", "y2"])
        return data.x1[0], data.y1[0], data.x2[0], data.y2[0]

    def frame_to_video(self):
        im_list = sorted(os.listdir(self.cover_s_r_path))
        img = Image.open(os.path.join(self.cover_s_r_path, im_list[0]))
        img_size = img.size

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, img_size)

        for im_name in im_list:
            frame = cv2.imdecode(np.fromfile(os.path.join(self.cover_s_r_path, im_name), dtype=np.uint8), -1)
            video_writer.write(frame)

        video_writer.release()
        print('Video generation completed.')

    def recover_images(self):
        preprocess = T.Compose([
            T.CenterCrop(con.cropsize_val),
            T.ToTensor(),
        ])

        cover_list = sorted(os.listdir(self.im_dir))
        for num, filename in enumerate(cover_list, start=1):
            steg_image_pil = Image.open(os.path.join(self.im_dir, filename))
            steg_image = preprocess(steg_image_pil).unsqueeze(0).cuda()
            recovered_cover_image, recovered_secret_image = self.steg.recovery(steg_image)

            torchvision.utils.save_image(recovered_cover_image,
                                         os.path.join(self.recovered_cover_path, f"{filename[:-12]}_rec.png"))
            torchvision.utils.save_image(recovered_secret_image,
                                         os.path.join(self.recovered_secret_path, f"{filename[:-12]}_rec.png"))

            print(f'Recovered image {num}')

    def combine_images(self):
        secret_list = sorted(os.listdir(self.recovered_secret_path))
        cover_list = sorted(os.listdir(self.im_dir))

        for secret_file, cover_file in zip(secret_list, cover_list):
            xmin, ymin, xmax, ymax = self.read_txt(os.path.join(self.label_path, f"{secret_file[:-8]}.txt"))
            w, h = xmax - xmin, ymax - ymin

            transf = T.Compose([T.CenterCrop((h, w))])

            rec_pil = Image.open(os.path.join(self.recovered_secret_path, secret_file))
            rec_pil = transf(rec_pil)

            cov_pil = Image.open(os.path.join(self.im_dir, cover_file))
            box = (xmin, ymin, xmax, ymax)
            cov_pil.paste(rec_pil, box)

            final_path = os.path.join(self.cover_s_r_path, f"{secret_file[:-8]}_final.png")
            cov_pil.save(final_path)

    def process(self):
        self.remove_files(self.recovered_cover_path)
        self.remove_files(self.recovered_secret_path)
        self.remove_files(self.cover_s_r_path)

        self.recover_images()
        self.combine_images()
        self.frame_to_video()


if __name__ == '__main__':
    processor = recover_images(fps=25)
    processor.process()