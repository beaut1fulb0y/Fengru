import argparse
import platform

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ResnetDataset

'''
| task | option                        |
|------|------------------------------ |
|  0   | train on dataset              |
|  1   | train on view based dataset   |
|  2   | train for view classification | 
'''

Parser = argparse.ArgumentParser()
Parser.add_argument("-t", "--task", default=0, type=int, help="tasks, 0 to 2 are available")
Parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("-l", "--lr", default=0.001, type=float, help="learning rate")
Parser.add_argument("-e", "--epochs", default=100, type=int, help="training epochs")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-v", "--view", default=None, type=str, help="which view of image, 1, 2, 3 are available")


def train(model, dataloader, criterion, optimizer, device):
    print("training model")

    model.train()
    running_loss = 0
    correct = 0
    total = 0
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (data, labels) in process_bar:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


def validation(model, dataloader, criterion, optimizer, device):
    print("testing model")

    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (data, labels) in process_bar:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        running_loss += loss.item()
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


def train_on_dataset(args):
    print('creating dataset')
    if platform.system() == "Windows":
        if args.task == 0 or args.task == 2:
            train_data_dir = "data\\data2\\unlabeled"
            test_data_dir = "data\\data1\\unlabeled"
        elif args.task == 1:
            train_data_dir = "data\\data2\\unlabeled" + args.view
            test_data_dir = "data\\data1\\unlabeled" + args.view
        else:
            pass

    elif platform.system() == "Darwin":
        if args.task == 0 or args.task == 2:
            train_data_dir = "data/data2/unlabeled"
            test_data_dir = "data/data1/unlabeled"
        elif args.task == 1:
            train_data_dir = "data/data2/unlabeled" + args.view
            test_data_dir = "data/data1/unlabeled" + args.view
        else:
            pass
    else:
        raise OSError("not compatible (we have not tested yet)")

    train_dataset = ResnetDataset(train_data_dir, view=True if (args.task == 2) else False)
    test_dataset = ResnetDataset(test_data_dir, view=True if (args.task == 2) else False)
    print('creating dataloader')
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    # for idx, (data, labels) in enumerate(train_dataloader):
    #     print(data.shape)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        loss, accuracy = train(model, train_dataloader, criterion, optimizer, device)
        valid_loss, valid_accuracy = validation(model, test_dataloader, criterion, optimizer, device)
        print(
            f"Epoch {args.epochs + 1}/{epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}")
        writer.add_scalar("Training Loss", loss, epoch)
        writer.add_scalar("Training Accuracy", accuracy, epoch)
        writer.add_scalar("Validation Loss", valid_loss, epoch)
        writer.add_scalar("Validation Accuracy", valid_accuracy, epoch)
        save_root = args.save_path
        if not (epoch % 20):
            save_path = f"{save_root}/{str(epoch)}.pth"
            torch.save(model.state_dict(), save_path)

        writer.close()


if __name__ == "__main__":
    args = Parser.parse_args()
    train_on_dataset(args)
