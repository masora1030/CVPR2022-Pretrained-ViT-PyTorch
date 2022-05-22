""" Summary utilities
Modification of timm's code.

Title: pytorch-image-models
Author: Ross Wightman
Date: 2021
Availability: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/summary.py
"""

import csv
from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass


def original_update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    if train_metrics:
        rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
