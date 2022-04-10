
import os 
import numpy as np
import pandas as pd
from PIL import Image

os.makedirs('test_images', exist_ok=True)

points = np.random.uniform(0, 1, size=(100, 2))
actions = np.random.randint(0, 4, size=(100,))

images = np.random.randint(0, 255, size=(100, 32, 32)).astype(np.uint8)
paths = []

for i in range(images.shape[0]):
    img = Image.fromarray(images[i])
    img.save(os.path.join('test_images', f'{i}.png'), format='PNG')
    paths.append(os.path.join(f'{i}.png'))
    
data = pd.DataFrame({'x1': points[:, 0].reshape(-1), 'x2': points[:, 1].reshape(-1), 'action': actions, 'impath': paths})
data.to_csv('sample_data.csv')