import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import models


# make dataset for pytorch

class leaf_dataset(Dataset):
    def __init__(self, data, label):
        super(leaf_dataset, self).__init__()
        self.X = data
        self.Y = label
        
    def __getitem__(self, idx):
        self.x = self.X[idx]
        self.y = self.Y[idx]
        return torch.FloatTensor(self.x), torch.FloatTensor(self.y)
    
    def __len__(self):
        return len(self.X)

img_path = "./train/"
label_path = "./label.csv"
test_img_path = "./test/"

label = pd.read_csv(label_path, index_col=0)
y = label.label.values # csv파일에서 label의 모든 값을 불러온다
y = np.expand_dims(y, axis=1)

# train폴더의 jpg파일을 읽어서 리스트에 추가
imgs_list = sorted(glob.glob(img_path+"*.jpg"))
img_to_numpy = []

for img_path in imgs_list:
    img = Image.open(img_path) # image open
    img = np.array(img).transpose(2,0,1) # img numpy 변환 
    img = img / 255.0 # img 0 ~ 1 로 scale 조정 / rgb 값이 0~255이기 때문에 위와같이 나누어 준다
    img_to_numpy.append(img) # 변환이 완료된 이미지를 배열에 추가
    
img_to_numpy = np.array(img_to_numpy)

X_train, X_valid, y_train, y_valid = train_test_split(img_to_numpy, y, test_size = 0.2, stratify=y, random_state=22)
# 학습과 검증데이터를 분리 / stratify=y는 y의 비율에 맞게 분배하는것으로 학습과 검증데이터에 쏠려서 분배되는 현상을 방지
    
train_set = leaf_dataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

valid_set = leaf_dataset(X_valid, y_valid)
valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

leaf_classification = models.resnet18(pretrained=True)

for param in leaf_classification.parameters():
  param.requires_grad = False

# change the output layer
num_classes = 1
num_ftrs = leaf_classification.fc.in_features
# resnet의 마지막 층에 두 가지 결과만을 나타내도록 층 추가
leaf_classification.fc = nn.Sequential(nn.Linear(num_ftrs, 20),
                                     nn.Dropout(0.4),
                                     nn.Linear(20,1)
                                    )

optimizer = optim.Adam(leaf_classification.parameters(), lr = 1e-3)
loss_fn = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5, verbose=True)

best_loss = 9999
for epoch in range(5):
    leaf_classification.train()
    train_loss = []
    train_acc_count = 0
    for data, label in train_loader:
        output = torch.sigmoid(leaf_classification(data))
        loss = loss_fn(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        
        output = torch.where(output > 0.5, 1, 0)
        train_acc_count += (output == label).sum()
        
    with torch.no_grad():
        leaf_classification.eval()
        valid_loss = []
        valid_acc_count = 0
        
        for data, label in valid_loader:
            output = torch.sigmoid(leaf_classification(data))
            loss = loss_fn(output, label)
            
            valid_loss.append(loss.item())
            
            output = torch.where(output > 0.5, 1, 0)
            valid_acc_count += (output == label).sum()
    scheduler.step(np.mean(valid_loss))
    if best_loss > np.mean(valid_loss):
        best_loss = np.mean(valid_loss)
        torch.save(leaf_classification.state_dict(), "best_resnet_model.pth")
    print(f"EPOCH : {epoch}, loss : {np.mean(train_loss)}, acc : {train_acc_count / len(train_set)} val_loss : {np.mean(valid_loss)}, val_acc : {valid_acc_count / len(valid_set)}")