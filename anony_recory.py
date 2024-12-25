import cv2
import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

# 添加模块搜索路径
sys.path.append(".")
sys.path.append("/home/lhb/grf")
sys.path.append("/home/lhb/grf/lama")
sys.path.append("lama/configs")
sys.path.append("lama/models")
sys.path.append("/home/lhb/grf/HiNet")
sys.path.append("/home/lhb/grf/yolov7")
sys.path.append("yolov7/fast_reid_master")

from HiNet.steganography import Steganography
from HiNet import config as con

class RecoverImages:
    """
    A class for recovering images and generating videos from steganographic images.
    """
    def __init__(self, fps=30, im_dir="example/lama_s_pic"):
        """
        Initialize the RecoverImages class.

        :param fps: Frames per second for the output video.
        :param im_dir: Directory containing the input images.
        """
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
        """
        Remove all files with specific extensions from a directory.

        :param path: Directory path to remove files from.
        """
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return
        for file in os.listdir(path):
            if file.endswith(('.png', '.jpg', '.txt')):
                try:
                    os.remove(os.path.join(path, file))
                except Exception as e:
                    print(f"Failed to remove {file}: {e}")
        print(f"Files removed from {path}")

    @staticmethod
    def read_txt(file_path):
        """
        Read coordinates from a text file.

        :param file_path: Path to the text file containing coordinates.
        :return: A tuple of (x1, y1, x2, y2) coordinates.
        """
        try:
            data = pd.read_csv(file_path, sep=' ', names=["x1", "y1", "x2", "y2"])
            return data.x1.iloc[0], data.y1.iloc[0], data.x2.iloc[0], data.y2.iloc[0]
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None

    def frame_to_video(self):
        """
        Generate a video from a sequence of images.
        """
        im_list = sorted(os.listdir(self.cover_s_r_path))
        if not im_list:
            print("No images found to create video.")
            return
        img = Image.open(os.path.join(self.cover_s_r_path, im_list[0]))
        img_size = img.size

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, img_size)

        for im_name in im_list:
            frame = cv2.imdecode(np.fromfile(os.path.join(self.cover_s_r_path, im_name), dtype=np.uint8), -1)
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Failed to decode {im_name}")
        video_writer.release()
        print('Video generation completed.')

    def recover_images(self):
        """
        Recover the cover and secret images from steganographic images.
        """
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
        """
        Combine recovered secret images with cover images based on coordinates.
        """
        secret_list = sorted(os.listdir(self.recovered_secret_path))
        cover_list = sorted(os.listdir(self.im_dir))

        for secret_file, cover_file in zip(secret_list, cover_list):
            xmin, ymin, xmax, ymax = self.read_txt(os.path.join(self.label_path, f"{secret_file[:-8]}.txt"))
            if xmin is None:
                continue
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
        """
        The main processing function that orchestrates the recovery and video generation.
        """
        self.remove_files(self.recovered_cover_path)
        self.remove_files(self.recovered_secret_path)
        self.remove_files(self.cover_s_r_path)

        self.recover_images()
        self.combine_images()
        self.frame_to_video()

if __name__ == '__main__':
    processor = RecoverImages(fps=25)
    processor.process()
