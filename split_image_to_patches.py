import torchvision, torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

img = torchvision.io.read_image('spectrogram2.png')
print(img.shape)
reference_image  = torchvision.io.read_image('spectrogram.png')

# Get the target height and width from the reference image
target_height, target_width = reference_image.shape[1], reference_image.shape[2]

# Resize the source image to the target size
img = F.interpolate(img.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False)

# Remove the batch dimension
img = img.squeeze(0).permute(1, 2, 0)

print(img.shape)

H, W, C = img.shape

patch_width = 15
n_rows = H // patch_width
n_cols = W // patch_width

cropped_img = img[:n_rows * patch_width, :n_cols * patch_width, :]

#
# Into patches
# [n_rows, n_cols, patch_width, patch_width, C]
#
patches = torch.empty(n_rows, n_cols, patch_width, patch_width, C)
for chan in range(C):
    patches[..., chan] = (
        cropped_img[..., chan]
        .reshape(n_rows, patch_width, n_cols, patch_width)
        .permute(0, 2, 1, 3)
    )
    
#
#Plot
#
# n_rows = 5
# n_cols = 1
f, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5))

import random
import numpy as np
for row_idx in range(n_rows):
    for col_idx in range(n_cols):
        # if random.random() < 0.8:
        if False:
            gray = np.zeros(patches[row_idx, col_idx, ...].shape)
            gray[..., -1] = 1
            gray[..., 0] = 128/255
            gray[..., 1] = 128/255
            gray[..., 2] = 128/255
            axs[row_idx, col_idx].imshow(gray)
        else:
            axs[row_idx, col_idx].imshow(patches[row_idx, col_idx, ...] / 255)
            # axs[row_idx].imshow(patches[row_idx, col_idx, ...] / 255)

# for row_idx in range(n_rows):
#     for col_idx in range(n_cols):
#         r_idx = random.randint(0, patches.shape[0]-1)
#         c_idx = random.randint(0, patches.shape[1]-1)
#         axs[row_idx].imshow(patches[r_idx, c_idx, ...] / 255)

for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
f.subplots_adjust(wspace=0.05, hspace=0.05)
# f.subplots_adjust(wspace=0.05, hspace=0.53)

plt.savefig('patches.png')
