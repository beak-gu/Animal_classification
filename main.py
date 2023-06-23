## load library
import numpy as np
import json
from PIL import Image
import PIL.Image as pilimg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import Subset, dataloader


## parameter
is_Test = False
# is_Test = True
num_epochs = 25
batch_size = 128

data_path = r"C:\Users\ngw77\Desktop\Ncloud\Image_Training\Dataset"
save_path = r"C:\Users\ngw77\Desktop\Ncloud\Image_Training\output"

data_train_path = os.path.join(data_path, "train")
data_valid_path = os.path.join(data_path, "valid")
data_test_path = os.path.join(data_path, "test")

# 이미지 tensor형태로 변환
transform_function = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 모델 입력사이즈로 resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = {}
dataset["train"] = datasets.ImageFolder(data_train_path, transform_function)
dataset["valid"] = datasets.ImageFolder(data_valid_path, transform_function)
dataset["test"] = datasets.ImageFolder(data_test_path, transform_function)

print(
    "data proportion(train:valid:test) = %s : %s : %s"
    % (len(dataset["train"]), len(dataset["valid"]), len(dataset["test"]))
)

## data loader 선언
# gpu있을 시에는 4*gpu갯수 => num_workers=4
dataloaders, batch_num = {}, {}
dataloaders["train"] = torch.utils.data.DataLoader(
    dataset["train"], batch_size=batch_size, shuffle=True
)
dataloaders["valid"] = torch.utils.data.DataLoader(
    dataset["valid"], batch_size=batch_size, shuffle=False
)
dataloaders["test"] = torch.utils.data.DataLoader(
    dataset["test"], batch_size=batch_size, shuffle=False
)

batch_num["train"], batch_num["valid"], batch_num["test"] = (
    len(dataloaders["train"]),
    len(dataloaders["valid"]),
    len(dataloaders["test"]),
)

print(
    "batch_size : %d,  number of batch(train/valid/test) : %d / %d / %d"
    % (batch_size, batch_num["train"], batch_num["valid"], batch_num["test"])
)


def imshow(inp, title=None):
    # imshow in tensor
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.numpy([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    # pause a bit so that plots are updated
    
    num_show_img = 8
    
    class_names = 