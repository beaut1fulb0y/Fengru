import os

import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import CustomResNet18
from utils import visualize_heatmap


def predict(path_list: list):
    model_sep = CustomResNet18()
    load_path = os.path.join('parameters', 'best.pth')
    state_dict = torch.load(load_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model_sep.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
    ])

    pred_list = []
    heatmap_list = []

    for path in path_list:
        image = Image.open(path)
        image_input = transform(image).unsqueeze(0)
        out = F.softmax(model_sep(image_input)).squeeze()
        view = torch.argmax(out).item()

        pred, heatmap = visualize_heatmap(view, image)
        pred_list.append(pred)
        heatmap_list.append(heatmap)

    # 1 for afflicted 0 for unafflicted
    sum_pred = torch.tensor([0, 0, 0])
    for i in pred_list:
        sum_pred += i

    ave_pred = sum_pred / len(pred_list)
    status = torch.argmax(ave_pred).item()
