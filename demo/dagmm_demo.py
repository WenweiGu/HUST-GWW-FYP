import os
import sys
import time


sys.path.append("../")
from networks.dagmm.dagmm import DAGMM
from common.data_preprocess import generate_windows, preprocessor
from common.dataloader import load_dataset
import tensorflow as tf
from common.evaluation import evaluate_all
from common.evaluation import compute_salience

dataset = "SMD"
subdataset = "machine-1-2"
compression_hiddens = [20, 5]
compression_activation = tf.nn.tanh
estimation_hiddens = [20, 10]
estimation_activation = tf.nn.tanh
estimation_dropout_ratio = 0.25
minibatch = 1024
epoch = 2
lr = 0.0001
lambdaone = 0.01
lambdatwo = 0.0001
normalize = True
random_seed = 123
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    # preprocessing
    pp = preprocessor()
    data_dict = pp.normalize(data_dict)

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    start = time.time()

    dagmm = DAGMM(
        comp_hiddens=compression_hiddens,
        comp_activation=compression_activation,
        est_hiddens=estimation_hiddens,
        est_activation=estimation_activation,
        est_dropout_ratio=estimation_dropout_ratio,
        minibatch_size=minibatch,
        epoch_size=epoch,
        learning_rate=lr,
        lambda1=lambdaone,
        lambda2=lambdatwo,
        normalize=normalize,
        random_seed=random_seed,
    )

    # predict anomaly score
    dagmm.fit(x_train)
    anomaly_score = dagmm.predict_prob(x_test)
    anomaly_label = x_test_labels

    end = time.time()

    time = end -start

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
    salience = compute_salience(anomaly_score, anomaly_label)
    print('time')
    print('   ', time)
    print('salience')
    print('   ', salience)