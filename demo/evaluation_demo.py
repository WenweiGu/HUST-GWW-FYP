import sys

sys.path.append("../")

import numpy as np
from common.evaluation import evaluate_all
from common.utils import pprint

if __name__ == "__main__":
    num_points = 100
    anomaly_label = np.random.choice([0, 1], size=num_points)
    anomaly_score = np.random.uniform(0, 1, size=num_points)
    anomaly_score_train = np.random.uniform(0, 1, size=num_points)

    metrics_iter, metrics_evt, theta_iter, theta_evt = evaluate_all(
        anomaly_score, anomaly_label, anomaly_score_train
    )
