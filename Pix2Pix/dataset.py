import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

        self.mode = mode
        self.root = root
        
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if self.mode == "train":
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        if "facades" in self.root:
            return {"A": img_A, "B": img_B}
        elif "cityscapes" in self.root:
            return {"A": img_A, "B": img_B}
        elif "maps" in self.root:
            return {"A": img_B, "B": img_A}
        elif "edges2shoes" in self.root:
            return {"A": img_B, "B": img_A}
        else:
            return {"A": img_A, "B": img_B}

        
    def __len__(self):
        return len(self.files)