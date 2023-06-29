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


# model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device="cpu",
    num_epochs=25,
    is_test=False,
    save_path="output",
    use_multi_gpu=True,
):
    since = time.time()

    # 저장 폴더 생성
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # 가장 좋은 모델 저장
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_f1 = 0.0, 0.0
    train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    # 각각의 에포크에는 training 과 validation set이 있다.
    for epoch in range(num_epochs):
        print("\n Epoch : %d/%d" % (epoch, num_epochs - 1))
        print("------------------------------")
        for phase in ["train", "valid"]:
            # 모델을 훈련 모드로 설정하라는 pytorch 라이브러리
            # 드롭아웃, 배치노말라이제이션, 그라디엔트계산, 매개변수업데이트...=> 훈련모드에서만 동작
            if phase == "train":
                model.train()
            # 모델을 훈련 모드로 설정하라는 pytorch 라이브러리
            # 드롭아웃 모듈은 평가 동안에 비활성화되어 예측의 일관성을 높이는 데 도움
            else:
                model.eval()

            running_loss, running_correct, num_cnt = 0.0, 0, 0
            pred_list, label_list = [], []

            for batch_idx, batch in enumerate(dataloaders[phase]):
                # for test
                if is_test:
                    # class의 수가 2이므로 2보다 크면 break
                    if batch_idx > 6:
                        break
                    # inputs = [128,3,224,224]
                    # labels = [128], tensor[1,1,1,0,1,1, ..]
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                # optimizer : 계산된 그라디언트를 기반으로 모델의 매개변수 업데이트 하는 최적화 '함수'
                # 기계 학습의 목표 : 모델의 예측 값 간의 불일치를 측정하는 손실 함수를 최소화 하는 것
                # 모델의 각 매개변수에 대해 기울기를 누적 => 기울기를 적용하여, 매개 변수를 업데이트 하기 전에 각 매개변수에 대해 기울기 0으로 설정
                optimizer.zero_grad()
                # torch.set_grad_enabled() => pythorch 함수 코드 블럭에 대해 그라디언트를 계산 할지 말지 지정 가능
                # with구문 : with 자원의 획득 as 자원의 반납 : 자원의 사용
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # outputs : (batch_size,num_classes)
                    _, preds = torch.max(outputs, 1)
                    # 예측된 출력과 실제 레이블 간의 차이점
                    # 모델의 예측이 Ground Truth와 얼마나 잘 일치하는지
                    loss = criterion(outputs, labels)
                    # 학습시만 backpropagation, backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)
                num_cnt += len(labels)
                print("///////////////////////////////////////////////////")
                print("Num Cnt: ", num_cnt)
                pred_list += preds.data.cpu().numpy().tolist()
                label_list += labels.data.cpu().numpy().tolist()
            if phase == "train":
                scheduler.step()
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_correct.double() / num_cnt).cpu * 100)
            epoch_f1 = float(f1_score(label_list, pred_list, average="mecro") * 100)
            if phase == "train":
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_f1.append(epoch_f1)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                valid_f1.append(epoch_f1)
            print(
                "{} Loss : {:.2f} | Acc : {:.2f} | f1 : {:.2f}".format(
                    phase, epoch_loss, epoch_acc, epoch_f1
                )
            )
            # save best model, validation acc가 높을때 저장, deep copy the model
            # if (phase == 'valid') and (epoch_acc > best_acc):
            if (phase == "valid") and (epoch_f1 > best_f1):
                best_idx = epoch
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                print(
                    "==> best model saved - %d | %.2f | %.2f"
                    % (best_idx, best_acc, best_f1)
                )

        time_elapsed = time.time() - since
        print(
            "\n\nTraining complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best valid Acc: %d - %.2f | %.2f" % (best_idx, best_acc, best_f1))

        # load best model weights
        model.load_state_dict(best_model_wts)

        if use_multi_gpu:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        weight_path = os.path.join(
            save_path, "model_%d_%.2f_%.2f.pt" % (best_idx, best_acc, best_f1)
        )
        torch.save(model_state_dict, weight_path)

        print("save model_%d_%.2f_%.2f.pt" % (best_idx, best_acc, best_f1))
    return (
        model,
        best_idx,
        best_acc,
        train_loss,
        train_acc,
        train_f1,
        valid_loss,
        valid_acc,
        valid_f1,
        weight_path,
    )
