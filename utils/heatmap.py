import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from model import ResNet18GradCAM, compute_gradcam


def visualize_heatmap(view, image):
    # image_np = np.asarray(image)
    model = ResNet18GradCAM()
    state_dict = torch.load(f'../parameters/best{view}.pth', map_location='cpu')
    model.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
    ])

    image_np = np.asarray(transforms.Resize((224, 224))(image))

    input_image = transform(image).unsqueeze(0)
    heatmap = compute_gradcam(model, input_image).squeeze(0)

    heatmap = np.transpose(heatmap.numpy(), (1, 2, 0))
    heatmap = np.squeeze(heatmap)

    plt.imshow(image_np)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.savefig(f'../runs/{view}.png')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../dataset/mydata.csv')
    view = 1
    df = df[(df['view'] == view) & (df['afflict'] == 'afflicted')]
    x = df.sample()
    x_list = list(map(str, df.sample().iloc[0].to_list()))
    path = os.path.join(*x_list)
    image = Image.open('../' + path)
    visualize_heatmap(view, image)
