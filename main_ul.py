import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import IntactDataset
from model import CustomResNet18

'''
| task | option                        |
|------|------------------------------ |
|  0   | train on dataset              |
|  1   | train on view based dataset   |
|  2   | train for view classification | 
'''

# set environment parser
Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
Parser.add_argument("-e", "--epochs", default=50, type=int, help="training epochs")
Parser.add_argument("-l", "--lr", default=0.0005, type=float, help="learning rate")
Parser.add_argument("-p", "--pretrain", default=True, type=bool, help="pretrained on Image Net")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-t", "--task", default=0, type=int, help="tasks, 0 to 2 are available")
Parser.add_argument("-v", "--view", default=None, type=str, help="which view of image, 1, 2, 3 are available")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")


# general train process
def train(model, dataloader, criterion, optimizer, device, train):
    print("training model")
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0
    correct = 0
    total = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (data, labels) in process_bar:
        data, labels = data.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


def train_on_intact_dataset(args):
    print('creating dataset')
    if args.task == 0 or args.task == 2:
        data_dir1 = os.path.join('data', 'data1', 'labeled')
        data_dir2 = os.path.join('data', 'data2', 'labeled')
    else:
        data_dir1 = os.path.join('data', 'data1', 'labeled', args.view)
        data_dir2 = os.path.join('data', 'data2', 'labeled', args.view)

    dataset = IntactDataset(data_dir1, data_dir2, view=True if (args.task == 2) else False, aug=False)
    dataset_aug = IntactDataset(data_dir1, data_dir2, view=True if (args.task == 2) else False, aug=True)
    print('creating dataloader')

    # Define the sizes of the training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))

    # Use SubsetRandomSampler to split the dataset into three sets
    train_sampler = SubsetRandomSampler(range(train_size))
    val_sampler = SubsetRandomSampler(range(train_size, train_size + val_size))
    test_sampler = SubsetRandomSampler(range(train_size + val_size, len(dataset)))

    print(train_size)

    train_dataloader = DataLoader(dataset=dataset_aug, batch_size=args.batch_size,
                                  num_workers=args.num_workers, sampler=train_sampler)
    train_naug_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, sampler=val_sampler)
    test_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, sampler=test_sampler)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"

    if args.task == 0 or args.task == 1:
        num_classes = 2
    else:
        num_classes = 3

    model = CustomResNet18(num_classes, pretrain=args.pretrain, dropout=args.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    fc_params = list(model.resnet.fc.parameters())
    other_params = [param for name, param in model.resnet.named_parameters() if "fc" not in name]
    optimizer = optim.Adam([
        {"params": fc_params, "lr": args.lr},
        {"params": other_params, "lr": args.lr / 10},
    ])
    writer = SummaryWriter()

    best = 100
    train_ac = -1
    test_ac = -1
    best_epoch = -1
    best_ac = -1
    for epoch in range(args.epochs):
        loss, accuracy = train(model, train_dataloader, criterion, optimizer, device, True)
        naug_loss, naug_accuracy = train(model, train_naug_dataloader, criterion, optimizer, device, False)
        valid_loss, valid_accuracy = train(model, valid_dataloader, criterion, optimizer,
                                           device, False)
        test_loss, test_accuracy = train(model, test_dataloader, criterion, optimizer,
                                         device, False)

        print(
            f"Epoch {args.epochs}/{epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")
        writer.add_scalar("Training Loss", loss, epoch)
        writer.add_scalar("Training Accuracy", accuracy, epoch)
        writer.add_scalar("Validation Loss", valid_loss, epoch)
        writer.add_scalar("Validation Accuracy", valid_accuracy, epoch)
        writer.add_scalar("Test Loss", test_loss, epoch)
        writer.add_scalar("Test Accuracy", test_accuracy, epoch)

        save_root = args.save_path
        if best > valid_loss and valid_accuracy < naug_accuracy:
            best_epoch = epoch
            best = valid_loss
            train_ac = accuracy
            best_ac = valid_accuracy
            test_ac = test_accuracy
            if args.task == 1:
                save_path = os.path.join(save_root, f'best{args.view}.pth')
            else:
                save_path = os.path.join(save_root, 'best.pth')
            torch.save(model.state_dict(), save_path)

        writer.close()

    print(f"train_acc: {train_ac}, best val_acc: {best_ac}, test_acc: {test_ac}, best epoch: {best_epoch}")


if __name__ == "__main__":
    args = Parser.parse_args()
    train_on_intact_dataset(args)