import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import CustomResNet18

if __name__ == '__main__':
    dataset = TestDataset('../dataset/mydata.csv')
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1)
    model_sep = CustomResNet18(num_classes=3, pretrain=False)
    model = [CustomResNet18(num_classes=2, pretrain=False), CustomResNet18(num_classes=2, pretrain=False),
             CustomResNet18(num_classes=2, pretrain=False)]
    state_dict = []
    sep_state_dict = torch.load('../parameters/best.pth', map_location='cpu')
    model_sep.load_state_dict(sep_state_dict)
    for i in range(1, 4):
        dict = torch.load(f'../parameters/best{str(i)}.pth',
                          map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        state_dict.append(dict)
    for i in range(3):
        model[i].load_state_dict(state_dict[i])
        model[i].eval()

    out_list = []
    label_list = []
    pred_list = []

    for batch_idx, (data, label) in enumerate(dataloader):
        data = data.squeeze(dim=0)
        views = torch.argmax(softmax(model_sep(data)), dim=1)
        pred = torch.tensor([0, 0], dtype=torch.float32)
        for i in range(len(views)):
            view = views[i].item()
            out = model[view](data[i].unsqueeze(0))
            x = softmax(out).squeeze(dim=0)
            pred += x
        pred /= len(views)
        label_list.append(int(label[0][0]))
        pred_list.append(pred[1].item())
        out_list.append(1 if pred[1] > pred[0] else 0)

    out_list = np.array(out_list)
    label_list = np.array(label_list)
    pred_list = np.array(pred_list)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(out_list)):
        if label_list[i] == 1 and out_list[i] == 1:
            TP += 1
        elif label_list[i] == 0 and out_list[i] == 0:
            TN += 1
        elif label_list[i] == 0 and out_list[i] == 1:
            FP += 1
        else:
            FN += 1

    print(f"TP:{TP}, TF:{TN}, FP:{FP}, FN:{FN}")
    print(out_list)
    print(pred_list)
    print(label_list)
    np.save('../parameters/out_list.npy', out_list)
    np.save('../parameters/pred_list.npy', pred_list)
    np.save('../parameters/label_list.npy', label_list)
