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
from flask import *
from efficientnet_pytorch import EfficientNet


app = Flask(__name__)

weights_path = "output/model_2_100.00_100.00.pt"


@app.route("/upload", methods=["POST"])
def up_loadfile():
    return render_template("upload.html")


transform_function = transforms.Compose(
    [
        # transforms.Resize((224, 224)),  # 모델 입력사이즈로 resize => b0
        transforms.Resize((456, 456)),  # 모델 입력사이즈로 resize => b5
        transforms.ToTensor(),  # 0101로 바꾸기
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.route("/uploader", method=["GET", "POST"])
def uploader():
    if request.method == "POST":
        f = request.files["jpg"]
        model_load = EfficientNet.from_pretrained("efficientnet-b5", num_classes=6)
        state_dict = torch.load(weights_path, map_location="cpu")  # load weight
        model_load.load_state_dict(
            state_dict, strict=False
        )  # insert weight to model structure
        model_load = model_load.to("cpu")
        # 모델 테스트 할 거니까 evaluation 모드로 시작
        model_load.eval()
        transformed_f = transform_function(f)
        outputs = model_load(transformed_f)
        _, preds = torch.max(outputs, 1)
        return preds


if __name__ == "__main__":
    app.run(debug=True)
