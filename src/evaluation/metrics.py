"""常用指标封装。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def multiclass_logloss(y_true, prob_pred) -> float:
    return log_loss(y_true, prob_pred, labels=np.arange(prob_pred.shape[1]))
