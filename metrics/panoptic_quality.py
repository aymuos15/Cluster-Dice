import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import ast

from metrics.legacy_dice import dice

def create_match_dict(pred_label_cc, gt_label_cc):
    pred_to_gt = {}
    gt_to_pred = {}
    individual_matches = set()
    dice_scores = {}

    # Get unique labels
    pred_labels = np.unique(pred_label_cc)[1:]  # Exclude background (0)
    gt_labels = np.unique(gt_label_cc)[1:]  # Exclude background (0)

    for pred_label in pred_labels:
        pred_mask = pred_label_cc == pred_label
        for gt_label in gt_labels:
            gt_mask = gt_label_cc == gt_label
            
            # Check if there's any overlap
            if np.any(np.logical_and(pred_mask, gt_mask)):
                pred_item = pred_label.item()
                gt_item = gt_label.item()
                
                if pred_item not in pred_to_gt:
                    pred_to_gt[pred_item] = []
                if gt_item not in pred_to_gt[pred_item]:
                    pred_to_gt[pred_item].append(gt_item)
                
                if gt_item not in gt_to_pred:
                    gt_to_pred[gt_item] = []
                if pred_item not in gt_to_pred[gt_item]:
                    gt_to_pred[gt_item].append(pred_item)
    
    # Calculate Dice scores for combined masks
    for pred_item, gt_items in pred_to_gt.items():
        pred_mask = pred_label_cc == pred_item
        for combo in all_combinations(gt_items):
            gt_mask = np.zeros_like(gt_label_cc, dtype=bool)
            for gt_item in combo:
                gt_mask = np.logical_or(gt_mask, gt_label_cc == gt_item)
            dice_scores[(pred_item, tuple(combo))] = dice(pred_mask, gt_mask)

    for gt_item, pred_items in gt_to_pred.items():
        gt_mask = gt_label_cc == gt_item
        for combo in all_combinations(pred_items):
            pred_mask = np.zeros_like(pred_label_cc, dtype=bool)
            for pred_item in combo:
                pred_mask = np.logical_or(pred_mask, pred_label_cc == pred_item)
            dice_scores[(tuple(combo), gt_item)] = dice(pred_mask, gt_mask)
    
    # Add gt labels with no matches
    for gt_label in gt_labels:
        gt_item = gt_label.item()
        if gt_item not in gt_to_pred:
            gt_to_pred[gt_item] = []
    
    # Add pred labels with no matches
    for pred_label in pred_labels:
        pred_item = pred_label.item()
        if pred_item not in pred_to_gt:
            pred_to_gt[pred_item] = []
            
    return {"pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred, "individual_matches": individual_matches, "dice_scores": dice_scores}

def all_combinations(lst):
    return [c for i in range(1, len(lst) + 1) for c in combinations(lst, i)]

def get_all_matches(matches):
    match_df = pd.DataFrame(columns=["Prediction", "Ground Truth", "Dice"])

    for gt, preds in matches["gt_to_pred"].items():
        if not preds:  # If there are no predictions for this ground truth
            match_df = match_df._append(pd.DataFrame({
                "Prediction": ["[]"], 
                "Ground Truth": [gt], 
                "Dice": [0.0]
            }), ignore_index=True)
        else:
            for combo in all_combinations(preds):
                combo_tuple = tuple(combo)
                dice_score = matches["dice_scores"].get((combo_tuple, gt), 0.0)
                match_df = match_df._append(pd.DataFrame({
                    "Prediction": [str(list(combo))], 
                    "Ground Truth": [gt], 
                    "Dice": [dice_score]
                }), ignore_index=True)

    for pred, gts in matches["pred_to_gt"].items():
        if not gts:  # If there are no ground truths for this prediction
            match_df = match_df._append(pd.DataFrame({
                "Prediction": [pred], 
                "Ground Truth": ["[]"], 
                "Dice": [0.0]
            }), ignore_index=True)
        else:
            for combo in all_combinations(gts):
                combo_tuple = tuple(combo)
                dice_score = matches["dice_scores"].get((pred, combo_tuple), 0.0)
                match_df = match_df._append(pd.DataFrame({
                    "Prediction": [pred], 
                    "Ground Truth": [str(list(combo))], 
                    "Dice": [dice_score]
                }), ignore_index=True)
        
    return match_df

def remove_single_element_lists(x):
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            lst = ast.literal_eval(x)
            if isinstance(lst, list) and len(lst) == 1:
                return lst[0]
        except:
            pass
    return x

def optimal_matching(df):
    # Expand the DataFrame to handle lists in 'Ground Truth'
    expanded_data = []
    for i, row in df.iterrows():
        ground_truths = row['Ground Truth'] if isinstance(row['Ground Truth'], list) else [row['Ground Truth']]
        for gt in ground_truths:
            expanded_data.append({'Prediction': row['Prediction'], 'Ground Truth': gt, 'Dice': row['Dice']})

    expanded_df = pd.DataFrame(expanded_data)

    # Create the cost matrix
    unique_predictions = expanded_df['Prediction'].unique()
    unique_ground_truths = expanded_df['Ground Truth'].unique()

    cost_matrix = np.ones((len(unique_predictions), len(unique_ground_truths)))

    for i, pred in enumerate(unique_predictions):
        for j, gt in enumerate(unique_ground_truths):
            match = expanded_df[(expanded_df['Prediction'] == pred) & (expanded_df['Ground Truth'] == gt)]
            if not match.empty:
                cost_matrix[i, j] = 1 - match.iloc[0]['Dice']

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract the optimal matching
    optimal_matching = []
    for i, j in zip(row_ind, col_ind):
        pred = unique_predictions[i]
        gt = unique_ground_truths[j]
        match = expanded_df[(expanded_df['Prediction'] == pred) & (expanded_df['Ground Truth'] == gt)]
        if not match.empty:
            optimal_matching.append(match.iloc[0])

    return optimal_matching

def is_list_with_multiple_elements(x):
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            lst = ast.literal_eval(x)
            return isinstance(lst, list) and len(lst) > 1
        except:
            return False
    return False

def pq(pred_label_cc_gpu, gt_label_cc_gpu):
    matches = create_match_dict(pred_label_cc_gpu, gt_label_cc_gpu)
    match_df = get_all_matches(matches)

    # In the pq function, after applying remove_single_element_lists:
    match_df = match_df[~match_df["Ground Truth"].apply(is_list_with_multiple_elements)]

    fp = [pred for pred in matches["pred_to_gt"] if not matches["pred_to_gt"][pred]]
    fn = [gt for gt in matches["gt_to_pred"] if not matches["gt_to_pred"][gt]]

    #drop the false positives and false negatives
    match_df = match_df[~match_df["Prediction"].isin(fp)]
    match_df = match_df[~match_df["Ground Truth"].isin(fn)]
    match_df['Prediction'] = match_df['Prediction'].apply(remove_single_element_lists)
    match_df['Ground Truth'] = match_df['Ground Truth'].apply(remove_single_element_lists)

    optimal_matches = optimal_matching(match_df)
    optimal_matching_df = pd.DataFrame(optimal_matches)

    tp = optimal_matching_df['Prediction'].unique()

    rq = len(tp) / (len(tp) + 0.5 * len(fp) + 0.5 * len(fn))
    sq = sum(optimal_matching_df['Dice']) / len(tp)
    pq = rq * sq

    return pq