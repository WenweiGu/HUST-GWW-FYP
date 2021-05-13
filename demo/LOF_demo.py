import sys
import logging
from pyod.models.lof import LOF
import time

sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluate_all
from common.evaluation import compute_salience

dataset = "SMD"
subdataset = "machine-1-2"
n_neighbors = 20
leaf_size = 30
p = 2   # Parameter for the Minkowski
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
    )

    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    # data preprocessing for MSCRED
    start = time.time()
    od = LOF(n_neighbors=n_neighbors, leaf_size=leaf_size, p=p)
    od.fit(x_train)

    # get outlier scores
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
