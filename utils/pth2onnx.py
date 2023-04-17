import torch.onnx

from model import CustomResNet18

if __name__ == '__main__':
    model = CustomResNet18(num_classes=2, pretrain=False)
    model_statedict = torch.load('../parameters/best1.pth', map_location='cpu')
    model.load_state_dict(model_statedict)
    model.eval()

    input_data = torch.randn(10, 3, 224, 224, device='cpu')

    input_name = ['input']
    output_name = ['output']

    torch.onnx.export(model, input_data, '../parameters/ResNet18.onnx', verbose=True, input_names=input_name, output_names=output_name)
