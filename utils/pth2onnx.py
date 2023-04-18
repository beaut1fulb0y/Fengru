import os

import torch.onnx

from model import CustomResNet18

if __name__ == '__main__':
    model = CustomResNet18(num_classes=2, pretrain=False)
    path = os.path.join('..', 'parameters', 'best.pth')
    model_statedict = torch.load(path, map_location='cpu')
    model.load_state_dict(model_statedict)
    model.eval()

    input_data = torch.randn(10, 3, 224, 224, device='cpu')

    input_name = ['input']
    output_name = ['output']

    export_path = os.path.join('..', 'parameters', 'ResNet18.onnx')
    torch.onnx.export(model, input_data, export_path, verbose=True, input_names=input_name, output_names=output_name)
