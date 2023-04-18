import torch
import torch.nn.functional as F
from torch import Tensor

from model import ResNet18GradCAM


def compute_gradcam(model: ResNet18GradCAM, input_tensor: Tensor, target_class=None):
    model.eval()

    output = model(input_tensor)
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()

    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output, retain_graph=True)

    gradients = model.get_gradients()
    conv_output = model.get_conv_output()

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    heatmap = torch.mul(conv_output, weights).sum(dim=1, keepdim=True)

    heatmap = F.relu(heatmap)
    heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

    # Normalize the heatmap
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    return heatmap
