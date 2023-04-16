import torch
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
    label_list = [[], [], []]

    for i in range(3):
        for batch_idx, (data, label) in enumerate(dataloader[i]):
            out = model[i](data)
            x = softmax(out).squeeze(dim=0)
            out_class = 1 if x[1] > x[0] else 0
            out_list[i].append(out_class)
            label_list[i].append(int(label))
    for i in range(3):
        print(out_list[i])
    for i in range(3):
        print(label_list[i])


