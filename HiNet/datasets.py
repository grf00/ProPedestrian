import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            #self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
            self.cover_files = natsorted(sorted(glob.glob(c.TRAIN_PATH_COVER + "/*." + c.format_train)))
            self.secret_files = natsorted(sorted(glob.glob(c.TRAIN_PATH_SECRET + "/*." + c.format_train)))
        else:
            # test
            #self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))
            self.cover_files = natsorted(sorted(glob.glob(c.VAL_PATH_COVER + "/*." + c.format_val)))
            self.secret_files = natsorted(sorted(glob.glob(c.VAL_PATH_SECRET + "/*." + c.format_val)))



    def __getitem__(self, index):
        try:
            cover_image = Image.open(self.cover_files[index])
            secret_image = Image.open(self.secret_files[index])
            cover_image = to_rgb(cover_image)
            secret_image = to_rgb(secret_image)
            cover_item = self.transform(cover_image)
            secret_item = self.transform(secret_image)
            return cover_item, secret_item

        except:
            return self.__getitem__((index + 1) % len(self.cover_files))#递归调用 __getitem__ 方法，尝试加载下一个图像。

    def __len__(self):
        # if self.mode == 'shuffle':
        #     return max(len(self.files_cover), len(self.files_secret))
        #
        # else:
        return len(self.cover_files)


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# Training data loader
# trainloader = DataLoader(
#     Hinet_Dataset(transforms_=transform, mode="train"),
#     batch_size=c.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=8,
#     drop_last=True
# )
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    drop_last=True
)