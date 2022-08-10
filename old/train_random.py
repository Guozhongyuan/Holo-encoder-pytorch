'''
    train Unet on random size square
'''


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from model.v1 import Model


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



class random_dataset(Dataset):

    def __init__(self, num_file):
        super().__init__()
        self.file_list = []
        self.num = num_file

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        input = torch.zeros([1, 1, 1024, 1024]).cuda()
        cx = random.randint(512-128, 512+128)
        cy = random.randint(512-128, 512+128)
        l = random.randint(1, 8) * 16  # 16 - 128
        input[0, 0, cx-l:cx+l, cy-l:cy+l] = 1.0
        return input

    @staticmethod
    def collate_fn(batch):
        return torch.cat(batch, axis=0)



if __name__ == '__main__':
    
    model = Model().cuda()

    lr = 0.001

    dataset = random_dataset(20000)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
        drop_last=True)

    loss_max = np.inf
    for epoch in range(10):
        
        if epoch == 9:
            lr = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []
        for step, input in enumerate(tqdm(dataloader)):
            input = input.cuda()
            phase, image_norm, image_fft = model(input)
            mask = input!=0
            mse = (image_norm[0, 0, 384:640, 384:640] - image_fft[0, 0, 384:640, 384:640]).square()
            loss = mse.mean()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        loss_mean = np.mean(losses)
        print(loss_mean)
        if loss_mean < loss_max:
            loss_max = loss_mean
            torch.save(model.state_dict(), './ckpt/best.pth')

    # TODO add: distance infomation during fft, follow the real experiments (fixed distance 0.15m ?)
    # TODO check: matlab and pytorch should be the same
    # TODO image dataset
    # TODO loss function
    # TODO image only center region ?