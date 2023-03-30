import os
import platform

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


class ResnetDataset(Dataset):
    def __init__(self, root_path, view: bool):
        super(ResnetDataset).__init__()
        self.root_path = root_path
        self.file_list = path_to_id(root_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
        ])
        self.view = view

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_dir = self.file_list[idx]
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


class IntactDataset(Dataset):
    def __init__(self, root_path1, root_path2, view: bool):
        super(ResnetDataset).__init__()
        self.root_path1 = root_path1
        self.root_path2 = root_path2
        self.file_list1 = path_to_id(root_path1)
        self.file_list2 = path_to_id(root_path2)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
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
    # img = Image.open('..\\data\\data1\\labeled\\unafflicted\\labeled\\0\\0.jpg')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.14751036, 0.14716739, 0.14718017), (0.14871628, 0.14826333, 0.14827214)),
    ])
    dataset = IntactDataset(root_path1="../data/data1", root_path2="../data/data2", view=False)
    print(dataset[12])
