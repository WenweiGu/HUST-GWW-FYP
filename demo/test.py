import numpy as np
import logging
from common.dataloader import load_dataset
import sys
sys.path.append("../")

dataset = "SMD"
subdataset = "machine-1-1"


def correlate_normalize(series1, series2):
    correlate = float(np.correlate(series1, series2))
    norm1 = np.linalg.norm(series1)
    norm2 = np.linalg.norm(series2)

    if norm1 == 0 and norm2 == 0:
        return 0

    elif norm1 == 0:
        norm1 += 0.01

    elif norm2 == 0:
        norm2 += 0.01

    correlate = correlate / (norm1 * norm2)

    return correlate


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


def get_1d(x_data):
    weight = np.zeros(x_data.shape[1])
    for i in range(x_data.shape[1]):
        s = 0
        for j in range(x_data.shape[1]):
            s += abs(correlate_normalize(x_data[:, i], x_data[:, j]))
        weight[i] = s

    weight = weight / np.sum(weight)
    x_data = np.dot(x_data, weight)

    return x_data

print(np.arange(5).shape)
