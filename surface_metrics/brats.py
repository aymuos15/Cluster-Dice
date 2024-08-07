import numpy as np

# from metrics.legacy_dice import dice
from metrics.surface_dist import compute_surface_distances
from metrics.surface_dist import compute_robust_hausdorff

def lesion_wise_dice(pred_label_cc, gt_label_cc):

    tp = []
    fn = []
    fp = []
    gt_tp = []
    lesion_scores = []

    for gtcomp in range(np.max(gt_label_cc)):
        gtcomp += 1

        ## Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1
        
        ## Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp = pred_tmp*gt_tmp
        intersecting_cc = np.unique(pred_tmp) 
        intersecting_cc = intersecting_cc[intersecting_cc != 0] 
        for cc in intersecting_cc:
            tp.append(cc)

        ## Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp,intersecting_cc,invert=True)] = 0
        pred_tmp[np.isin(pred_tmp,intersecting_cc)] = 1

        ## Calculating Lesion-wise Dice and HD95
        # lesion_scores.append(dice(pred_tmp,gt_tmp))  
        lesion_dist = (compute_surface_distances(pred_tmp,gt_tmp, (1, 1, 1)))
        lesion_scores.append(compute_robust_hausdorff(lesion_dist, 95))

        ## Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(
            pred_label_cc[np.isin(
                pred_label_cc,tp+[0],invert=True)])
    
    # lesion_dice = sum(lesion_scores)/(len(lesion_scores) + len(fp))
    lesion_dists = (np.sum(lesion_scores) + len(fp)*374 ) /  (len(lesion_scores) + len(fp))
    
    return lesion_dists