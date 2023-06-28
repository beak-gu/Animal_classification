## load library
import numpy as np
import json
from PIL import Image
import PIL.Image as pilimg
from tensorboard import summary
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
from efficientnet_pytorch import EfficientNet
import train


## parameter
is_Test = False
# is_Test = True
num_epochs = 25
################## batch_size가 클수록 좋기는 한데,,, 일단 컴터가 안좋으니까 64 랩실컴으로 돌릴때는 128으로 하자
batch_size = 64

data_path = r"C:\Users\ngw77\Desktop\Ncloud\Dataset_AI"
save_path = r"C:\Users\ngw77\Desktop\Ncloud\Image_Training\output"

data_train_path = os.path.join(data_path, "train")
data_valid_path = os.path.join(data_path, "valid")
data_test_path = os.path.join(data_path, "test")

# 이미지 tensor형태로 변환
transform_function = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 모델 입력사이즈로 resize
        transforms.ToTensor(),  # 0101로 바꾸기
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


dataset = {}
dataset["train"] = datasets.ImageFolder(data_train_path, transform_function)
dataset["valid"] = datasets.ImageFolder(data_valid_path, transform_function)
dataset["test"] = datasets.ImageFolder(data_test_path, transform_function)

print(dataset["train"])
print("////////")

print(
    "data proportion(train:valid:test) = %s : %s : %s"
    % (len(dataset["train"]), len(dataset["valid"]), len(dataset["test"]))
)
print("////////")


## data loader 선언
#################### gpu있을 시에는 4*gpu갯수 => num_workers=4
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
print("////////")


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

    class_names = {"0": "Gorani", "1": "Noru"}

    input, classes = next(iter(dataloader["train"]))
    out = torchvision.utils.make_grid(input[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

    input, classes = next(iter(dataloader["valid"]))
    out = torchvision.utils.make_grid(input[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

    input, classes = next(iter(dataloader["test"]))
    out = torchvision.utils.make_grid(input[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])


## batch의 tensor 이미지를 확인하기 위한 함수
def check_image_from_tensor(check_image, check_class):
    title = list(check_class.cpu().numpy())  # torch tensor to list

    # 5x1 형식으로 만들기 위해
    num_image = len(title)
    if num_image <= 5:
        columns = num_image
        rows = 1
    else:
        columns = 5
        rows = int(np.ceil(num_image / columns))

    fig = plt.figure(figsize=(3 * columns, 4 * rows))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(1, num_image + 1):
        inp = check_image[i - 1].numpy().transpose((1, 2, 0))
        inp = std * inp + mean  # 원본 이미지로 변환
        inp = np.clip(inp, 0, 1)

        fig.add_subplot(rows, columns, i)
        plt.title(title[i - 1])
        plt.imshow(inp)
    plt.show()


num_show_img = 10
# data check
inputs, classes = next(iter(dataloaders["train"]))
print(inputs.size())
############# inputs의 사이즈 : torch.Size([128, 3, 224, 224]) #####################
check_image, check_class = inputs[:num_show_img], classes[:num_show_img]
check_image_from_tensor(check_image, check_class)

model_name = "efficientnet-b0"  # b5
num_classes = 2  # 노루 고라니
freeze_extractor = True  # 과하게 학습하는 것을 방지 FC layer만 학습하고 efficientNet extractor 부분은 freeze하여 학습시간 단축, 89860 vs 4097408
use_multi_gpu = False
########################gpu 있는 곳에서 학습 하면 True로 바꿔주기

Image_size = EfficientNet.get_image_size(model_name)
print("model input shape : (%d x %d)" % (Image_size, Image_size))
model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

# 과하게 학습하는 것을 방지 FC layer만 학습하고 efficientNet extractor 부분은 freeze하여 학습시간 단축, 89860 vs 4097408
if freeze_extractor:
    print("extractor freeeze")
    for n, p in model.named_parameters():
        if "_fc" not in n:
            p.requires_grad = False


# 파라미터의 갯수를 세는 함수
def count_parameters(model):
    total_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_trainable_params += p.numel()

    return total_trainable_params


# define optimizer, criterion
criterion = nn.CrossEntropyLoss()  # 분류이므로 cross entrophy 사용

# optimizer 선언, SGD, Adam으로도 해보자
# optimizer = optim.SGD(model.parameters(),
#                          lr = 0.05,
#                          momentum=0.9,
#                          weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98739)  # LR 스케쥴러, 점점 줄어든다
