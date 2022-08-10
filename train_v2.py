'''
    UNet predict phase on IMG
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import torch
from tqdm import tqdm
import random
import pickle
import zlib
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model.v2 import Model, N, M

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class NumpyDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_list = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        tmp = self.data_list[idx]
        data = pickle.loads(zlib.decompress(tmp)).astype(np.float32)
        input = np.zeros([1, 1, N, M]).astype(np.float32)
        input[0, 0] = cv2.resize(data, (N, M))
        # label = np.zeros([1, 1, N, M]).astype(np.float32)
        # label[0, 0] = cv2.resize(data, (N, M))
        label = input
        return input, label
    
    @staticmethod
    def collate_fn(batch):
        inputs = np.concatenate([item[0] for item in batch], axis=0)
        labels = np.concatenate([item[1] for item in batch], axis=0)
        inputs = torch.tensor(inputs).cuda()
        labels = torch.tensor(labels).cuda()
        return inputs, labels
    

class Loss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm([N, M], elementwise_affine=False)
        
    def forward(self, reconstruction, label):
        target = self.norm(reconstruction)
        label = self.norm(label)
        npcc = -1 * (target * label).mean()
        return npcc



if __name__ == '__main__':
    
    model = Model().cuda()

    epochs = 12
    batch_size = 4
    lr = 1e-4

    trainset = NumpyDataset('/data/gzy/anime/anime_train.pkl')
    valset = NumpyDataset('/data/gzy/anime/anime_val.pkl')

    # trainset.data_list = trainset.data_list[0:800]
    # valset.data_list = valset.data_list[0:200]

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        pin_memory=False,
        drop_last=True)

    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        pin_memory=False,
        drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = Loss().cuda()
    loss_max = np.inf
    
    for epoch in range(epochs):
        
        model.train()
        loss_train = []
        for inputs, labels in tqdm(trainloader):
            phase, reconstruction = model(inputs)
            loss = loss_fn(reconstruction, labels)
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        loss_val = []
        with torch.no_grad():
            for inputs, labels in tqdm(valloader):
                phase, reconstruction = model(inputs)
                loss = loss_fn(reconstruction, labels)
                loss_val.append(loss.item())

        loss_mean = np.mean(loss_val)
        print('epoch', epoch, 'loss', loss_mean)
        
        if loss_mean < loss_max:
            loss_max = loss_mean
            torch.save(model.state_dict(), './ckpt/best.pth')