import logging
import os
import sys
import traceback
import hydra
import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

def lama_pre():
    default_lama_path = 'E:/study/python/Study-myself/person-anonymous/lama/big-lama/big-lama.pt'
    default_lama = torch.jit.load(default_lama_path, map_location=torch.device('cpu'))
    default_lama.eval()

    in_img_path = 'test-image/frame76.jpg'
    in_mask_path = 'test-image/frame76_mask.png'
    outdir= 'out-put/'

    in_img = cv2.imread(in_img_path)[:, :, ::-1] / 255.0
    in_mask = cv2.imread(in_mask_path) / 255.0

    in_img = torch.from_numpy(in_img).unsqueeze(0).permute(0, 3, 1, 2).type(torch.float)
    in_mask = torch.from_numpy(in_mask).unsqueeze(0).permute(0, 3, 1, 2).type(torch.float)
    in_img = in_img * (1 - in_mask)
    with torch.no_grad():
       # default_input = torch.cat([in_img, in_mask], dim=1)
        default_lama_result = default_lama(in_img,in_mask)

    default_lama_result = default_lama_result.permute(0, 2, 3, 1).squeeze().numpy()
    cv2.imwrite(os.path.join(outdir, 'default_lama.png'), default_lama_result[:, :, ::-1] * 255.0)

if __name__ == "__main__":
    lama_pre()