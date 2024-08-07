import numpy as np

def dice(im1, im2):

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum()) / (im1.sum() + im2.sum())

# Taken from: https://github.com/rachitsaluja/BraTS-2023-Metrics/blob/main/metrics.py