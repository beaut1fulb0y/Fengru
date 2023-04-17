import torch
from model import CustomResNet18

def get_input_image():
    pass

if __name__ == '__main__':
    # load models
    model = [CustomResNet18(), CustomResNet18(), CustomResNet18()]
    for i in range(1, 4):
        state_dict = torch.load(f'parameters/best{i}.pth')
        model[i].load_state_dict(state_dict)
        model[i].eval()

