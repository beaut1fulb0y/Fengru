import os

import numpy as np
from PIL import Image
from torchvision import transforms


def accumulate_images_data(root_dir):
    num_images = 0
    sum_pixel_values = np.zeros(3)
    sum_square_pixel_values = np.zeros(3)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file[-4: -1] + file[-1] == '.jpg':
                img = Image.open(os.path.join(subdir, file))
                img_t = transforms.Resize((224, 224))(img)
                img_t = transforms.ToTensor()(img_t)
                num_images += 1
                sum_pixel_values += img_t.view(3, -1).sum(axis=1).numpy()
                sum_square_pixel_values += (img_t.view(3, -1) ** 2).sum(axis=1).numpy()

    return num_images, sum_pixel_values, sum_square_pixel_values


root_dir = os.path.join('..', 'data')
num_images, sum_pixel_values, sum_square_pixel_values = accumulate_images_data(root_dir)

mean = sum_pixel_values / (num_images * 224 * 224)
stddev = np.sqrt((sum_square_pixel_values / (num_images * 224 * 224) - mean ** 2))

print(mean)
print(stddev)
