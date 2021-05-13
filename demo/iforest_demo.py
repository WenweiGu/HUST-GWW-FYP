import sys
import logging
from pyod.models.iforest import IForest
import time
sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from common.evaluation import evaluate_all
from common.evaluation import compute_salience

dataset = "SMD"
subdataset = "machine-1-2"
n_estimators = 1
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    start = time.time()

    od = IForest(n_estimators=n_estimators)

    od.fit(x_train)

    anomaly_score = od.decision_function(x_test)

    anomaly_label = x_test_labels

    end = time.time()

    time = end - start

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
    salience = compute_salience(anomaly_score, anomaly_label)
    print('time')
    print('   ', time)
    print('salience')
    print('   ', salience)