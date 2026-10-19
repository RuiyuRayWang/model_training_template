import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def evaluate_with_thresholds(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.asarray(thresholds)
    
    if y_true.shape != y_prob.shape:
        raise ValueError(f'y_true shape {y_true.shape} != y_prob shape {y_prob.shape}')
    if y_true.shape[1] != thresholds.shape[0]:
        raise ValueError(
            f'Threshold count {thresholds.shape[0]} must match label count {y_true.shape[1]}'
        )
    
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    
    metrics = {
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'samples_f1': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'exact_match': float(np.mean(np.all(y_pred == y_true, axis=1))),
    }
    
    pr_auc_per_label = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        if np.unique(yj).size < 2:
            pr_auc_per_label.append(np.nan)
            continue
        pr_auc_per_label.append(average_precision_score(yj, y_prob[:, j]))
    
    metrics['macro_pr_auc'] = float(np.nanmean(pr_auc_per_label))
    return metrics, np.array(pr_auc_per_label), y_pred

# Example usage once val/test arrays are available:
# val_metrics, val_pr_auc_per_label, val_pred = evaluate_with_thresholds(
#     val_targets, val_probs, selected_label_thresholds
# )
# test_metrics, test_pr_auc_per_label, test_pred = evaluate_with_thresholds(
#     test_targets, test_probs, selected_label_thresholds
# )
# print('Validation metrics:', val_metrics)
# print('Test metrics:', test_metrics)


def find_best_threshold_per_label(y_true, y_prob, grid=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError(f'y_true shape {y_true.shape} != y_prob shape {y_prob.shape}')
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    
    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=float)
    best_f1 = np.zeros(n_labels, dtype=float)
    
    for j in range(n_labels):
        yj = y_true[:, j]
        pj = y_prob[:, j]
        if np.unique(yj).size < 2:
            thresholds[j] = 0.5
            best_f1[j] = np.nan
            continue
        
        local_best_thr = 0.5
        local_best_f1 = -1.0
        for thr in grid:
            pred = (pj >= thr).astype(int)
            score = f1_score(yj, pred, zero_division=0)
            if score > local_best_f1:
                local_best_f1 = score
                local_best_thr = thr
        
        thresholds[j] = local_best_thr
        best_f1[j] = local_best_f1
    
    return thresholds, best_f1