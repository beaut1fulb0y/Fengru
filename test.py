import torch
import numpy as np
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import CustomResNet18

if __name__ == '__main__':
    dataset = [TestDataset('dataset/mydata.csv', i) for i in range(1, 4)]
    dataloader = [DataLoader(dataset=dataset[i], batch_size=1, num_workers=1) for i in range(3)]
    model = [CustomResNet18(num_classes=2, pretrain=False), CustomResNet18(num_classes=2, pretrain=False),
             CustomResNet18(num_classes=2, pretrain=False)]
    state_dict = []
    for i in range(1, 4):
        dict = torch.load(f'parameters/best{str(i)}.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        state_dict.append(dict)
    for i in range(3):
        model[i].load_state_dict(state_dict[i])
        model[i].eval()

    out_list = [[], [], []]
    pred_list = [[], [], []]
    label_list = []

    for i in range(3):
        for batch_idx, (data, label) in enumerate(dataloader[i]):
            out = model[i](data)
            x = softmax(out).squeeze(dim=0)
            pred_list[i].append(float(x[1]))
            out_class = 1 if x[1] > x[0] else 0
            out_list[i].append(out_class)
            if i == 0:
                label_list.append(int(label))


    out_list = np.array(out_list)
    label_list = np.array(label_list)
    pred_list = np.array(pred_list)

    np.save('parameters/out_list3.npy', out_list)
    np.save('parameters/pred_list3.npy', pred_list)

    out_list = np.sum(out_list, axis=0)
    pred_list = np.sum(pred_list, axis=0) / 3

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(out_list)):
        if label_list[i] == 1 and out_list[i] > 1:
            TP += 1
        elif label_list[i] == 0 and out_list[i] < 2:
            TN += 1
        elif label_list[i] == 0 and out_list[i] > 1:
            FP += 1
        else:
            FN += 1

    print(f"TP:{TP}, TF:{TN}, FP:{FP}, FN:{FN}")
    print(out_list)
    print(pred_list)
    print(label_list)
    np.save('parameters/out_list.npy', out_list)
    np.save('parameters/pred_list.npy', pred_list)
    np.save('parameters/label_list.npy', label_list)
