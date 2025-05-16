from u_attention import *
import torch.optim as optim
from datasat_stage2 import Dataset
import numpy as np
import os
import random
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


def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def focal_loss(logit, target, gamma=2, alpha=0.25):
    n, c, h, w = logit.size()
    criterion = nn.CrossEntropyLoss()
    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    return loss


def train_epoch(epoch, model, optimizer, criteron, l1, train_loader, show_interview=3):
    model.train()
    loss_all, count = 0, 0
    for step, [Data_patch, GT] in enumerate(tqdm(train_loader)):
        for i in Data_patch.keys():
            Data_patch[i] = Data_patch[i].to(device).type(torch.float32)
        GT = GT.to(device).type(torch.float32)

        optimizer.zero_grad()
        out = model(Data_patch)
        loss = criteron(out, GT.long())
        loss.backward()
        optimizer.step()
        loss_all = loss_all + loss.item()
        count = count + 1
    return float(loss_all / count)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(max_epoch, batchsz, lr):
    set_seed(0)

    train_data = Dataset(root + "RGB_Norm.mat", root + "SAR_Norm.mat", root + "GT.mat", data='Data_1', mode='train',channel=3)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = CD_Net().to(device)

    load = False
    if load ==True:
        checkpoint = torch.load('../CD_model/Data_1/best.mdl')
        print("min_loss:", checkpoint['best_val'])
        model.load_state_dict(checkpoint['state_dict'])
        print('Load success')
    else:
        print('No model load')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteron = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    best_loss = 10
    for epoch in range(max_epoch):
        epoch = epoch+50
        train_loss = train_epoch(epoch, model, optimizer, criteron, l1, train_dataloader)
        if epoch % 25 == 0:
            state = dict(epoch=epoch + 1, state_dict=model.state_dict(), best_val=train_loss)
            torch.save(state, "../CD_model/Data_1/best_{:d}.mdl".format(epoch))
        if train_loss <= best_loss:
            state = dict(epoch=epoch + 1, state_dict=model.state_dict(), best_val=train_loss)
            torch.save(state, '../CD_model/Data_1/best.mdl')
            best_loss = train_loss
        print("epoch: %d  best_loss = %.7f train_loss = %.7f" % (epoch + 1, best_loss, train_loss))


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
    train(100, 64, 0.0001)
    model = CD_Net().to(device)
