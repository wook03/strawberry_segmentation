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
from torchvision import models

img_path = "./train/"
label_path = "./label.csv"
test_img_path = "./test/"

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

# 존재하는 가중치를 토대로 모델사용
leaf_classification.load_state_dict(torch.load("best_resnet_model.pth"))

# 이미지들을 불러들이는 과정
img_list = sorted(glob.glob("./test/*.jpg"))
img_to_numpy = []

for path in img_list:
    img = Image.open(path) # 이미지들의 크기를 128*128로서 
    img = img.resize((200,200))
    img = np.array(img).transpose(2,0,1) # (1, 128, 128)
    img = img / 255.0 
    img_to_numpy.append(img)    
test_data = torch.FloatTensor(img_to_numpy)

leaf_classification.eval()
with torch.no_grad():
    output_prob = torch.sigmoid(leaf_classification(test_data))
    print(output_prob.numpy().flatten())

    output = torch.where(output_prob > 0.5,1,0)
    print(output.numpy().flatten())


rows, columns = 5, 5
fig, axes = plt.subplots(rows, columns, figsize=(15,30))
output_text = ["Normal" if output[i]==0 else "Abnormal" for i in range(rows*columns)]
for row in range(rows):
    for column in range(columns):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.set_title(f"predict : {output_text[row*columns+column]}\n {output_prob[row*columns+column].numpy()}")
        axis.imshow(img_to_numpy[row*columns+column].transpose(1,2,0))
plt.show()
