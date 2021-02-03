import random
import torch
import numpy as np
from PIL import Image

"""
自作データローダ、遅い
"""


class DataLoaderSelf:

    def __init__(self, file_list: list, label_list: list, batch_size: int, channel: int = 3, shuffle: bool = False):
        self.file_list = file_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.channel = channel
        if shuffle:
            _zip = list(zip(file_list, label_list))
            random.shuffle(_zip)
            self.file_list, self.label_list = zip(*_zip)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        used_data_no = self._iter * self.batch_size
        if used_data_no >= len(self.file_list):
            raise StopIteration()

        img_path = self.file_list[used_data_no: (self._iter+1)*self.batch_size]
        label = self.label_list[used_data_no: (self._iter+1)*self.batch_size]

        img = np.array([np.array(Image.open(i)) for i in img_path])

        # normalize
        img = img/255

        # z score
        r_mean = 0.43
        r_std = 0.22
        g_mean = 0.5
        g_std = 0.22
        b_mean = 0.31
        b_std = 0.2
        img[:, :, :, 0] = (img[:, :, :, 0] - r_mean) / r_std
        img[:, :, :, 1] = (img[:, :, :, 1] - g_mean) / g_std
        img[:, :, :, 2] = (img[:, :, :, 2] - b_mean) / b_std

        # batch first
        img = img.transpose(0, 3, 1, 2)

        self._iter += 1
        return torch.tensor(img), torch.tensor(label)
