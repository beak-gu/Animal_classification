# two = save model_19_99.12_99.12.pt
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


def model_load_def(weights_path):
    model_name = "efficientnet-b0"  # b5
    num_classes = 2  # 장싱, 비정상
    freeze_extractor = True  # FC layer만 학습하고 efficientNet extractor 부분은 freeze하여 학습시간 단축, 89860 vs 4097408
    use_multi_gpu = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_load = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)  # load weight
    model_load.load_state_dict(
        state_dict, strict=False
    )  # insert weight to model structure
    model_load = model_load.to(device)
    # 모델 테스트 할 거니까 evaluation 모드로 시작
    model_load.eval()
    criterion = nn.CrossEntropyLoss()  # 분류이므로 cross entrophy 사용

    return model_load, criterion, device


def model_test(model, dataloader, device, criterion):
    def imshow(inp, title=None):
        # Imshow for Tensor.
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    class_names = [
        "non_pig",
        "pig",
    ]
    model.eval()
    running_loss, running_correct, num_cnt = 0.0, 0, 0
    pred_list = []
    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            print(int(torch.max(probabilities[0]) * 100), "% 확률로 판단되었습니다.")
            _, preds = torch.max(outputs, 1)
            pred_list += preds.data.cpu().numpy().tolist()
    return pred_list


if __name__ == "__main__":
    class_names = {0: "멧돼지가 아닙니다", 1: "멧돼지 입니다"}
    weights_path = (
        r"C:\Users\ngw77\Desktop\Ncloud\Image_Training\output\model_19_99.12_99.12.pt"
    )
    data_path = r"C:\Users\ngw77\Desktop\Ncloud\Dataset_AI\PIG"
    data_test_path = os.path.join(data_path, "test")
    transform_function = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(data_test_path, transform_function)
    dataloaders = {
        "test": torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
    }

    model_load, criterion, device = model_load_def(weights_path)
    pred_list = model_test(model_load, dataloaders["test"], device, criterion)
    i = 0
    for pred in pred_list:
        if pred == 1:
            i += 1
        class_name = class_names[pred]
        print("Prediction:", class_name)
    print("전체 Acuraccy :", int(i / len(pred_list) * 100))
