import os
import logging
from typing import Union
from PIL import Image
import numpy as np
import torch
import cv2

LAMA_MODEL_PATH = "lama/big-lama/big-lama.pt"


def load_jit_model(
    model_path: str,
    device: Union[torch.device, str]
) -> torch.jit._script.RecursiveScriptModule:
    logging.info(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location=device).to(device)
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        exit(-1)
    model.eval()
    return model


class LaMa:
    name = "lama"
    pad_mod = 8

    def __init__(self, device: Union[torch.device, str], **kwargs) -> None:
        self.device = device
        self.model = load_jit_model(LAMA_MODEL_PATH, device)

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: RGB IMAGE
        """
        dtype = image.dtype
        image = norm_img(image)
        mask = norm_img(mask if np.max(mask) > 1.0 else mask * 2)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255)
        return cur_res.astype(dtype)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: RGB IMAGE
        """
        dtype = image.dtype
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(image, mod=self.pad_mod)
        pad_mask = pad_img_to_modulo(mask, mod=self.pad_mod)

        result = self.forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        mask = mask[:, :, np.newaxis]
        mask = mask / 255 if np.max(mask) > 1.0 else mask
        result = result * mask + image * (1 - mask)
        return result.astype(dtype)


def norm_img(np_img: np.ndarray) -> np.ndarray:
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


if __name__ == "__main__":
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lama_model = LaMa(device)
    image_path = 'test-image/frame76.jpg'
    mask_path = 'test-image/frame76_mask.png'
    in_img = cv2.imread(image_path)
    # Load your image and mask here
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    image_np = np.array(image)[..., :3]
    mask_np = np.array(mask)
    masked = (image_np * (1 - mask_np[..., np.newaxis] / 255.0)).astype(np.uint8)

    # Perform inpainting
    #result = lama_model(image, mask)
    lama_image =lama_model(masked, mask_np)
    Image.fromarray(lama_image).show()
    #display(Image.fromarray(lama_image))

    # Example of saving result
    # np.save("result.npy", result)