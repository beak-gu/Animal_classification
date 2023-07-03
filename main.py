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
from train import train_model
from test import *

## parameter
is_Test = False
# is_Test = True
num_epochs = 25
################## batch_size가 클수록 좋기는 한데,,, 일단 컴터가 안좋으니까 64 랩실컴으로 돌릴때는 128으로 하자
batch_size = 128

data_path = r"~/Animal_dataset"
save_path = r"~/Image_Training-1/output"
weights_path = r"output/model_2_100.00_100.00.pt"

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
    dataset["train"], batch_size=batch_size, shuffle=True, num_workers=4
)
dataloaders["valid"] = torch.utils.data.DataLoader(
    dataset["valid"], batch_size=batch_size, shuffle=False, num_workers=4
)
dataloaders["test"] = torch.utils.data.DataLoader(
    dataset["test"], batch_size=batch_size, shuffle=False, num_workers=4
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
    # 0_반달가슴곰, 1_청설모, 2_다람쥐, 3_고라니, 4_멧돼지, 5_멧토끼
    class_names = {
        "0": "bear",
        "1": "cheang",
        "2": "daram",
        "3": "gorani",
        "4": "pig",
        "5": "rabbit",
    }

    input, classes = next(iter(dataloaders["train"]))
    out = torchvision.utils.make_grid(input[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

    input, classes = next(iter(dataloaders["valid"]))
    out = torchvision.utils.make_grid(input[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

    input, classes = next(iter(dataloaders["test"]))
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
# print(dataloaders["train"])
inputs, classes = next(iter(dataloaders["train"]))
print(inputs.size())
############# inputs의 사이즈 : torch.Size([128, 3, 224, 224]) #####################
check_image, check_class = inputs[:num_show_img], classes[:num_show_img]
check_image_from_tensor(check_image, check_class)

model_name = "efficientnet-b0"  # b5
num_classes = 6  # 0_반달가슴곰, 1_청설모, 2_다람쥐, 3_고라니, 4_멧돼지, 5_멧토끼
freeze_extractor = True  # 과하게 학습하는 것을 방지 FC layer만 학습하고 efficientNet extractor 부분은 freeze하여 학습시간 단축, 89860 vs 4097408
use_multi_gpu = True
########################gpu 있는 곳에서 학습 하면 True로 바꿔주기

Image_size = EfficientNet.get_image_size(model_name)
print("model input shape : (%d x %d)" % (Image_size, Image_size))
# 사전에 훈련된 eddicientnet data를 가져와 model에 저장
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# multi gpu(2개 이상)를 사용하는 경우
if use_multi_gpu:
    num_gpu = torch.cuda.device_count()
    if (device.type == "cuda") and (num_gpu > 1):
        print("use multi gpu : %d" % (num_gpu))
        model = nn.DataParallel(model, device_ids=list(range(num_gpu)))

model = model.to(device)

# define optimizer, criterion
criterion = nn.CrossEntropyLoss()  # 분류이므로 cross entrophy 사용

# optimizer 선언, SGD, Adam으로도 해보자
# optimizer = optim.SGD(model.parameters(),
#                          lr = 0.05,
#                          momentum=0.9,
#                          weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# print(dataloaders["train"]) _ 가능
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98739)  # LR 스케쥴러, 점점 줄어든다
# print(dataloaders["train"]) _ 가능


# train!!
(
    model,
    best_idx,
    best_acc,
    train_loss,
    train_acc,
    train_f1,
    valid_loss,
    valid_acc,
    valid_f1,
    weights_path,
) = train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device=device,
    num_epochs=num_epochs,
    is_test=is_Test,
    save_path=save_path,
    use_multi_gpu=use_multi_gpu,
)

## 결과 그래프 그리기
print(
    "best model : %d - %1.f / %.1f"
    % (best_idx, valid_acc[best_idx], valid_loss[best_idx])
)
print(
    "Best model valid Acc: %d - %.2f | %.2f | %.2f"
    % (best_idx, valid_acc[best_idx], valid_f1[best_idx], valid_loss[best_idx])
)
fig, ax1 = plt.subplots()

ax1.plot(train_acc, "b-")
ax1.plot(valid_acc, "r-")
plt.plot(best_idx, valid_acc[best_idx], "ro")
ax1.set_xlabel("epoch")
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel("acc", color="k")
ax1.tick_params("y", colors="k")

ax2 = ax1.twinx()
ax2.plot(train_loss, "g-")
ax2.plot(valid_loss, "k-")
plt.plot(best_idx, valid_loss[best_idx], "ro")
ax2.set_ylabel("loss", color="k")
ax2.tick_params("y", colors="k")

fig.tight_layout()
plt.show()

model_load, criterion, device = model_load_def(weights_path)
label_list, pred_list = model_test(
    model=model_load,
    dataloader=dataloaders["test"],
    device=device,
    criterion=criterion,
)
for i in range(1, 10 + 1):
    print(label_list, pred_list)
