import os
import platform

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def path_to_id(root):
    root_list = []
    for root_dir, _, files in os.walk(root, topdown=True):
        for file in files:
            if file[-4: -1] + file[-1] == '.jpg':
                file_path = os.path.join(root_dir, file)
                root_list.append(file_path)

    return (root_list)


class TestDataset(Dataset):
    def __init__(self, csv_path):
        super(TestDataset, self).__init__()
        self.csv_path = csv_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
        ])
        self.df = pd.read_csv(self.csv_path)
        self.df1 = self.df[
            (self.df['folder_id'] == 'data1') & (self.df['label'] == 'labeled')]
        self.df2 = self.df[
            (self.df['folder_id'] == 'data2') & (self.df['label'] == 'labeled')]

        self.patient_afflicted_num1 = self.df1[self.df1['afflict'] == 'afflicted']['patient'].max() + 1
        self.patient_unafflicted_num1 = self.df1[self.df1['afflict'] == 'unafflicted']['patient'].max() + 1
        self.patient_afflicted_num2 = self.df2[self.df2['afflict'] == 'afflicted']['patient'].max() + 1
        self.patient_unafflicted_num2 = self.df2[self.df2['afflict'] == 'unafflicted']['patient'].max() + 1

        self.patient_num1 = self.patient_afflicted_num1 + self.patient_unafflicted_num1
        self.patient_num2 = self.patient_afflicted_num2 + self.patient_unafflicted_num2

    def __len__(self):
        return self.patient_num1 + self.patient_num2 - 1

    def __getitem__(self, idx):
        if idx < self.patient_afflicted_num1:
            folder_idx = idx
            df = self.df1[(self.df1['afflict'] == 'afflicted') & (self.df1['patient'] == folder_idx)]
        elif idx < self.patient_num1:
            folder_idx = idx - self.patient_afflicted_num1
            df = self.df1[
                (self.df1['afflict'] == 'unafflicted') & (self.df1['patient'] == folder_idx)]
        elif idx < self.patient_num1 + self.patient_afflicted_num2:
            folder_idx = idx - self.patient_num1
            df = self.df2[(self.df2['afflict'] == 'afflicted') & (self.df2['patient'] == folder_idx)]
        else:
            folder_idx = idx - self.patient_num1 - self.patient_afflicted_num2
            df = self.df2[(self.df2['afflict'] == 'unafflicted') & (
                    self.df2['patient'] == folder_idx)]

        if df.empty:
            df = self.df1[(self.df1['afflict'] == 'afflicted') & (self.df1['patient'] == 0)]
            result = list(map(str, df.sample().iloc[0].to_list()))
            path = os.path.join(*result)
            img = Image.open(path)
            img = self.transform(img)
            label = 1 if 'afflicted' in result else 0

            return img, label

        img_tensor = []
        label_tensor = []

        for i in range(df.shape[0]):
            result = list(map(str, df.iloc[i].to_list()))
            path = os.path.join('../', *result)
            img = Image.open(path)
            img_input = self.transform(img)
            img_tensor.append(img_input)
            label = 1 if 'afflicted' in result else 0
            label_tensor.append(torch.tensor(label))

        img_tensor = torch.stack(img_tensor, dim=0)
        label_tensor = torch.tensor(label_tensor)

        return img_tensor, label_tensor


class IntactDataset(Dataset):
    def __init__(self, root_path1, root_path2, view: bool, aug: bool):
        super(IntactDataset).__init__()
        self.root_path1 = root_path1
        self.root_path2 = root_path2
        self.file_list1 = path_to_id(root_path1)
        self.file_list2 = path_to_id(root_path2)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(175, padding=4),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
        ]) if aug else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
        ])
        self.view = view

    def __len__(self):
        return len(self.file_list1) + len(self.file_list2)

    def __getitem__(self, idx):
        if idx < len(self.file_list1):
            # If the index is within the range of the first folder
            file_dir = self.file_list1[idx]

        else:
            file_dir = self.file_list2[idx - len(self.file_list1)]

        img = Image.open(file_dir)
        img = self.transform(img)
        if self.view:
            if platform.system() == "Windows":
                label = int(file_dir.split('\\')[3]) - 1
            elif platform.system() == "Darwin" or "Linux":
                label = int(file_dir.split('/')[3]) - 1
            else:
                raise OSError("not compatible (we have not tested yet)")

        else:
            if platform.system() == "Windows":
                label = 1 if '\\afflicted' in file_dir else 0
            elif platform.system() == "Darwin" or "Linux":
                label = 1 if '/afflicted' in file_dir else 0
            else:
                raise OSError("not compatible (we have not tested yet)")

        return img, label


if __name__ == "__main__":
    # test_dataset = ResnetDataset('..\\data')
    img = Image.open('../data/data1/labeled/1/afflicted/1/0.jpg')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(160, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=10),
    ])
    img = transform(img)
    img.show()
    dataset = IntactDataset(root_path1="../data/data1", root_path2="../data/data2", view=False, aug=True)
    print(dataset[12])
