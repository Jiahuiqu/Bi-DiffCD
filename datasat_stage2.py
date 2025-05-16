import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
import cv2 as cv
import os
import random



class Dataset(nn.Module):

    def __init__(self, name1, name2, REF, data, mode, channel=128, padding = 2, nums = 500):
        super(Dataset, self).__init__()
        self.channel = channel
        self.mode = mode
        self.filename_T1 = name1
        self.filename_T2 = name2
        self.REF = REF
        self.nums = nums
        self.padding = padding
        self.filename_T2_Mat = './mat_file/2000_1000_0_layer_f/'
        self.Data_all = {}
        if data == "Data_1":

            file = sorted(os.listdir(self.filename_T2_Mat))
            file = list(reversed(file))


            self.Data_all['RGB_f_step0_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step0_0']
            self.Data_all['RGB_f_step0_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step0_1']
            self.Data_all['RGB_f_step0_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step0_2']
            self.Data_all['RGB_f_step0_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step0_3']
            self.Data_all['RGB_f_step0_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step0_4']

            self.Data_all['RGB_f_step1000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step1000_0']
            self.Data_all['RGB_f_step1000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step1000_1']
            self.Data_all['RGB_f_step1000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step1000_2']
            self.Data_all['RGB_f_step1000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step1000_3']
            self.Data_all['RGB_f_step1000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step1000_4']
            self.Data_all['RGB_f_step1000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step1000_6']

            self.Data_all['RGB_f_step2000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step2000_0']
            self.Data_all['RGB_f_step2000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step2000_1']
            self.Data_all['RGB_f_step2000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step2000_2']
            self.Data_all['RGB_f_step2000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step2000_3']
            self.Data_all['RGB_f_step2000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step2000_4']
            self.Data_all['RGB_f_step2000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_f_step2000_6']
            self.Data_all['RGB_x_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['RGB_x_0']


            self.Data_all['SAR_f_step0_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step0_0']
            self.Data_all['SAR_f_step0_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step0_1']
            self.Data_all['SAR_f_step0_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step0_2']
            self.Data_all['SAR_f_step0_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step0_3']
            self.Data_all['SAR_f_step0_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step0_4']

            self.Data_all['SAR_f_step1000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step1000_0']
            self.Data_all['SAR_f_step1000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step1000_1']
            self.Data_all['SAR_f_step1000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step1000_2']
            self.Data_all['SAR_f_step1000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step1000_3']
            self.Data_all['SAR_f_step1000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step1000_4']
            self.Data_all['SAR_f_step1000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step1000_6']

            self.Data_all['SAR_f_step2000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step2000_0']
            self.Data_all['SAR_f_step2000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step2000_1']
            self.Data_all['SAR_f_step2000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step2000_2']
            self.Data_all['SAR_f_step2000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step2000_3']
            self.Data_all['SAR_f_step2000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step2000_4']
            self.Data_all['SAR_f_step2000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_f_step2000_6']
            self.Data_all['SAR_x_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['SAR_x_0']

            self.Data_all['RGB_x_org'] = loadmat(os.path.join(self.filename_T1))['RGB_Norm'][0:540, 111:921]
            self.Data_all['SAR_x_org'] = loadmat(os.path.join(self.filename_T2))['SAR_Norm'][0:540, 111:921]


            self.image_REF = loadmat(os.path.join(self.REF))['GT']
            self.image_REF = self.image_REF[0: 540, 111: 921]
            # self.image_REF[0:280, 110:400] = 3

        if data == "Data_2":
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['RGB_Norm']
            [row, col, bands] = self.image_T1.shape
            # self.image_T1 = self.image_T1.reshape(row * col, -1).transpose(1,0)

            self.image_T2 = loadmat(os.path.join(self.filename_T2))['SAR_Norm']
            # self.image_T2 = self.image_T2.reshape(row * col, -1).transpose(1, 0)

            self.image_REF = loadmat(os.path.join(self.REF))['GT']

        if data == "Data_3":
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['T1']
            [row, col, bands] = self.image_T1.shape
            # self.image_T1 = self.image_T1.reshape(row * col, -1).transpose(1,0)

            self.image_T2 = loadmat(os.path.join(self.filename_T2))['T2']
            # self.image_T2 = self.image_T2.reshape(row * col, -1).transpose(1, 0)

            self.image_REF = loadmat(os.path.join(self.REF))['Binary']
        self.h, self.w = self.Data_all['RGB_f_step0_0'].shape[0], self.Data_all['RGB_f_step0_0'].shape[1]
        for i in self.Data_all.keys():
            self.Data_all[i] = cv.copyMakeBorder(self.Data_all[i], self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)


        all_num = self.h * self.w
        random.seed(1)
        self.whole_point = self.image_REF.reshape(1, all_num)
        if self.mode == "train":
            self.Changed_point = random.sample(list(np.where(self.whole_point[0] == 255)[0]), int(len(np.where(self.whole_point[0] == 255)[0])*0.1))
            self.NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(len(np.where(self.whole_point[0] == 0)[0])*0.1))
            self.random_point = self.Changed_point+self.NChanged_point
        if self.mode == "test":
            self.random_point = list(range(all_num))

        self.device = torch.device("cuda:0")

    def __len__(self):
        return len(self.random_point)

    def __getitem__(self, index):

        original_i = int((self.random_point[index] / self.w))
        original_j = (self.random_point[index] - original_i * self.w)
        new_i = original_i + self.padding
        new_j = original_j + self.padding
        self.Data_patch = {}
        for i in self.Data_all.keys():
            self.Data_patch[i+'_patch'] = self.Data_all[i][new_i - self.padding: new_i + self.padding+1,
                               new_j - self.padding: new_j + self.padding+1, :].transpose(2, 0, 1).reshape(self.channel,self.padding * 2+1,self.padding * 2+1)

        GT = self.image_REF[original_i, original_j]/255

        return self.Data_patch, GT


# if __name__ == "__main__":
# pass