import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pandas as pd

import cc3d

from metrics.legacy_dice import dice 
from metrics.panoptic_quality import pq
from metrics.brats import lesion_wise_dice
from metrics.cluster_dice import proposed_dice

def plot(pred, gt_display):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(pred[0, :, :], cmap='gray')
    ax[0].set_title('Prediction')

    ax[1].imshow(gt_display[0, :, :], cmap='gray')
    ax[1].set_title('Ground Truth')

    combined = pred + gt_display

    ax[2].imshow(combined[0, :, :], cmap='rainbow')
    ax[2].set_title('Overlay (Prediction + Ground Truth)')

    legend_elements = [Patch(facecolor='purple', edgecolor='black', label='TN'),
                    Patch(facecolor='turquoise', edgecolor='black', label='FP'),
                    Patch(facecolor='yellow', edgecolor='black', label='FN'),
                    Patch(facecolor='red', edgecolor='black', label='TP')]

    ax[2].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.show()

def score_tally(pred, gt):

    cluster_dice = proposed_dice(pred, gt)

    pred_label_cc = cc3d.connected_components(pred)
    gt_label_cc = cc3d.connected_components(gt)

    brats_dice = lesion_wise_dice(pred_label_cc, gt_label_cc)
    pq_dice = pq(pred_label_cc, gt_label_cc)

    table = pd.DataFrame({
        'Metric': ['Legacy', 'BraTS', 'PQ', 'Proposed'],
        'Value': [dice(pred, gt), brats_dice, pq_dice, cluster_dice],
        # 'TP': ['N/A', brats_tp, pq_tp, 'N/A'],
        # 'FP': ['N/A', brats_fp, pq_fp, 'N/A'],
        # 'FN': ['N/A', brats_fn, pq_fn, 'N/A'],
    })

    print(table)