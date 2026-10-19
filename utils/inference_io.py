import os
import copy
import torch
import numpy as np


def make_split_config(base_cfg, split_name):
    cfg = copy.deepcopy(base_cfg)
    out_dir = cfg['dataset']['out_dir']
    split_file = os.path.join(out_dir, f'regions_and_topics_{split_name}.bed')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f'Missing split file: {split_file}')
    cfg['dataset']['bed_file'] = split_file
    cfg['dataset']['augment'] = False
    return cfg


def collect_predictions(model, loader, device):
    probs_all = []
    targets_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch['sequence'].to(device)
            y = batch['label'].float().to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            probs_all.append(probs.detach().cpu().numpy())
            targets_all.append(y.detach().cpu().numpy())
    return np.concatenate(targets_all, axis=0), np.concatenate(probs_all, axis=0)