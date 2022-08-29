import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class CustomDataAugmentation:
    def __init__(self, flip_chance=0.5, color_change_chance=0.10, gaussian_noise_chance=0.2, gaussian_noise_range=5.0, luminosity_changes_chance=0.125):
        self.flip_chance = flip_chance
        self.color_change_chance = color_change_chance
        self.gaussian_noise_chance = gaussian_noise_chance
        self.luminosity_changes_chance = luminosity_changes_chance
        self.gaussian_noise_range = gaussian_noise_range

    def __call__(self, x):
        pil_img = x

        if self.flip_chance is not None:
            # try horizontal flipping
            if random.random() < self.flip_chance:
                pil_img = TF.hflip(pil_img)

        if self.color_change_chance is not None and random.random() < self.color_change_chance:
            # transform color by using HUE
            pil_img = TF.adjust_hue(pil_img, (random.random() * 0.45 - 0.225))

        if self.gaussian_noise_chance is not None and random.random() < self.gaussian_noise_chance:
            # add gaussian noise
            img_np = np.asarray(pil_img).astype(np.float64)
            img_np += np.random.randn(img_np.shape[0], img_np.shape[1],
                                      img_np.shape[2]) * self.gaussian_noise_range
            img_np[img_np < 0] = 0
            img_np[img_np > 255] = 255
            pil_img = Image.fromarray(img_np.astype(np.uint8))

        if self.luminosity_changes_chance is not None and random.random() < self.luminosity_changes_chance:
            # Apply random changes that affect the luminosity and Sharpness of the image

            if np.random.randn() < 0:
                # lower brightness ... uniform ... from 0.75 to 1.0
                pil_img = TF.adjust_brightness(
                    pil_img, 1.0 - np.random.rand() * 0.25)
            else:
                # increase brightness ... uniform ... from 1.0 to 1.5
                pil_img = TF.adjust_brightness(
                    pil_img, 1.0 + np.random.rand() * 0.50)

            if np.random.randn() < 0:
                # lower contrast ... uniform ... from 0.50 to 1.0
                pil_img = TF.adjust_contrast(
                    pil_img, 1.0 - np.random.rand() * 0.5)
            else:
                # increase contrast ... uniform ... from 1.0 to 2.0
                pil_img = TF.adjust_contrast(
                    pil_img, 1.0 + np.random.rand() * 1.0)

            if np.random.randn() < 0:
                # lower gamma ... uniform ... from 0.50 to 1.0
                pil_img = TF.adjust_gamma(
                    pil_img, 1.0 - np.random.rand() * 0.50)
            else:
                # increase gamma ... uniform ... from 1.0 to 2.0
                pil_img = TF.adjust_gamma(
                    pil_img, 1.0 + np.random.rand() * 1.00)

            if np.random.randn() < 0:
                # lower the saturation ... uniform ... between 0.25 to 1.0 saturation
                pil_img = TF.adjust_saturation(
                    pil_img, 1.0 - np.random.rand() * 0.75)
            else:
                # increase the saturation ... uniform ... between 1.0 to 5.0
                pil_img = TF.adjust_saturation(
                    pil_img, 1.0 + np.random.rand() * 4.0)
        return pil_img
