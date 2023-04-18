import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import ResNet18GradCAM, compute_gradcam


def visualize_heatmap(view, image):
    model = ResNet18GradCAM()
    model.eval()
    load_path = os.path.join('..', 'parameters', f'best{view}.pth') if __name__ == '__main__' else os.path.join(
        'parameters', f'best{view}.pth')
    state_dict = torch.load(load_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
    ])

    image_np = np.asarray(transforms.Resize((224, 224))(image))

    input_image = transform(image).unsqueeze(0)
    out = model(input_image)
    predict = F.softmax(out).squeeze()

    heatmap = compute_gradcam(model, input_image).squeeze(0)

    heatmap = np.transpose(heatmap.numpy(), (1, 2, 0))
    heatmap = np.squeeze(heatmap)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    combined_image = np.float32(image_np) * 0.7 + np.float32(heatmap_colored) * 0.5
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    # cv2.imshow('img', combined_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(combined_image)
    save_path = os.path.join('..', 'runs', f'{view}.png') if __name__ == '__main__' else os.path.join('runs',
                                                                                                      f'{view}.png')
    plt.savefig(save_path)
    # plt.show()

    return predict, combined_image


if __name__ == '__main__':
    data_path = os.path.join('..', 'dataset', 'mydata.csv')
    df = pd.read_csv(data_path)
    view = 1
    df = df[(df['view'] == view) & (df['afflict'] == 'afflicted')]
    x = df.sample()
    x_list = list(map(str, df.sample().iloc[0].to_list()))
    path = os.path.join('..', *x_list)
    image = Image.open(path)

    visualize_heatmap(view, image)
