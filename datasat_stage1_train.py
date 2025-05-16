import numpy as np
import torch.nn as nn
from scipy.io import loadmat
import cv2 as cv
import os
import random


class Dataset(nn.Module):

    def __init__(self, name1, name2, REF, data, mode, channel=128, padding = 13, nums = 500):
        super(Dataset, self).__init__()
        self.channel = channel
        self.mode = mode
        self.filename_T1 = name1
        self.filename_T2 = name2
        self.REF = REF
        self.nums = nums
        self.padding = padding

        if data == "Data_1":
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['RGB_Norm']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['SAR_Norm']
            self.image_REF = loadmat(os.path.join(self.REF))['GT']

        self.padding_image_T1 = cv.copyMakeBorder(self.image_T1, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)
        self.padding_image_T2 = cv.copyMakeBorder(self.image_T2, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)
        self.h, self.w = self.image_T1.shape[0], self.image_T1.shape[1]
        all_num = self.h * self.w
        random.seed(1)
        self.whole_point = self.image_REF.reshape(1, all_num)
        if self.mode == "train":
            self.NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(all_num*0.1))
            self.random_point = self.NChanged_point
        if self.mode == "test":
            self.Changed_point = list(range(all_num))
            self.NChanged_point = list(range(all_num))
            self.random_point = self.Changed_point + self.NChanged_point

    def __len__(self):
        return len(self.random_point)

    def __getitem__(self, index):

        original_i = int((self.random_point[index] / self.w))
        original_j = (self.random_point[index] - original_i * self.w)
        new_i = original_i + self.padding
        new_j = original_j + self.padding


        RGB = self.padding_image_T1[new_i - self.padding: new_i + self.padding+1,
        new_j - self.padding: new_j + self.padding+1, :].transpose(2, 0, 1).reshape(self.channel,
                                                                                      self.padding * 2+1,
                                                                                      self.padding * 2+1)

        SAR = self.padding_image_T2[new_i - self.padding: new_i + self.padding+1,
        new_j - self.padding: new_j + self.padding+1, :].transpose(2, 0, 1).reshape(self.channel,
                                                                                      self.padding * 2+1,
                                                                                      self.padding * 2+1)

        GT = self.image_REF[original_i, original_j]

        return RGB, SAR, GT

# if __name__ == "__main__":
#     pass
