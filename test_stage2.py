from u_attention import *
from datasat_stage2 import Dataset
import numpy as np
import os
import warnings
import time
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
from tqdm import tqdm

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
device = torch.device("cuda:1")
warnings.filterwarnings("ignore")
root = '../data/data_1_RGB_SAR/'


def test(model):
    model.eval()
    checkpoint = torch.load('../CD_model/Data_1/best.mdl')
    print("min_loss:", checkpoint['best_val'])
    model.load_state_dict(checkpoint['state_dict'])
    test_data = Dataset(root + "RGB_Norm.mat", root + "SAR_Norm.mat", root + "GT.mat", data='Data_1', mode='test',
                        channel=3)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    if not os.path.exists('result'):
        os.makedirs('result')
    GT = loadmat('../data/data_1_RGB_SAR/GT.mat')['GT'][0: 540, 111: 921]
    H,W = GT.shape
    outimage = np.zeros((1, H * W))
    start = time.time()
    count = 0
    with torch.no_grad():
        for step, [Data_patch, GT] in enumerate(tqdm(test_dataloader)):
            for i in Data_patch.keys():
                Data_patch[i] = Data_patch[i].to(device).type(torch.float32)
            GT = GT.to(device).type(torch.float32)
            batch = GT.shape[0]
            logits = model(Data_patch)
            outimage[0, count:(count + batch)] = logits.argmax(dim=1).detach().cpu().numpy()
            count += batch
        outimage = outimage.reshape(H, W)
        filename = "../result/result.mat"
        savemat(filename, {"output": outimage})
        print("save success!!!!")
    end = time.time()
    print("running time:", end - start)


if __name__ == "__main__":

    model = CD_Net().to(device)
    test(model)
