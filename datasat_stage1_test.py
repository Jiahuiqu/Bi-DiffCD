import torch.nn as nn
from scipy.io import loadmat
import os

def Split_Patches(image):

    # 获取输入图像的宽度和高度
    width, height, band = image.shape
    patch_size = 27

    # 计算需要切分的行数和列数
    num_rows = height // patch_size
    num_cols = width // patch_size
    ALL_Patch = []
    # 循环切分图像并存储patch
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前patch的位置
            left = col * patch_size
            upper = row * patch_size
            right = left + patch_size
            lower = upper + patch_size

            patch = image[left: right, upper: lower, :]
            ALL_Patch.append(patch)
    return ALL_Patch

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

        if data == "Data_2":
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['RGB_Norm']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['SAR_Norm']
            self.image_REF = loadmat(os.path.join(self.REF))['GT']

        self.padding_image_T1 = self.image_T1[0:540, 111:921]
        self.padding_image_T2 = self.image_T2[0:540, 111:921]
        self.h, self.w = self.image_T1.shape[0], self.image_T1.shape[1]
        self.Patch_T1 = Split_Patches(self.padding_image_T1)
        self.Patch_T2 = Split_Patches(self.padding_image_T2)


    def __len__(self):
        return len(self.Patch_T2)

    def __getitem__(self, index):

        return self.Patch_T1[index].transpose(2,0,1), self.Patch_T2[index].transpose(2,0,1)

# if __name__ == "__main__":
#     pass
