import sys
import logging
import time
from networks import donut

sys.path.append("../")

import warnings
import numpy as np
from networks.donut import DonutTrainer, DonutPredictor
import tensorflow as tf
from networks.donut import Donut
from tensorflow import keras as k
from tfsnippet.modules import Sequential

from common.dataloader import load_dataset
from common.evaluation import evaluate_all
from common.evaluation import compute_salience

dataset = "SMD"
subdataset = "machine-1-2"
point_adjustment = True
iterate_threshold = True

warnings.filterwarnings("ignore")


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


if __name__ == "__main__":
    # Read the raw data.

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

    x_train_labels = np.zeros(x_train.shape[0])

    # If there is no label, simply use all zeros.

    # Split the training and testing data.
    # for CVAE
    train_values = get_1d(x_train)
    test_values = get_1d(x_test)
    # FOR MVAE
    # train_values = np.mean(x_train, axis=1)
    # test_values = np.mean(x_test, axis=1)
    train_labels, test_labels = x_train_labels, x_test_labels

    # Standardize the training and testing data.
    train_values, mean, std = donut.standardize_kpi(
        train_values, excludes=train_labels)
    test_values, _, _ = donut.standardize_kpi(test_values, mean=mean, std=std)

    # We build the entire model within the scope of `model_vs`,
    # it should hold exactly all the variables of `model`, including
    # the variables created by Keras layers.

    start = time.time()

    model = Donut(
        h_for_p_x=Sequential([
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            k.layers.Dense(100, kernel_regularizer=k.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )

    with tf.Session().as_default():
        trainer = DonutTrainer(model=model)
        predictor = DonutPredictor(model)

        trainer.fit(train_values, train_labels, mean, std)
        anomaly_score = predictor.get_score(test_values)
        anomaly_label = x_test_labels[-len(anomaly_score):]

        end = time.time()

        time = end - start

        evaluate_all(anomaly_score, anomaly_label)
        salience = compute_salience(anomaly_score, anomaly_label)
        print('time')
        print('   ', time)
        print('salience')
        print('   ', salience)
