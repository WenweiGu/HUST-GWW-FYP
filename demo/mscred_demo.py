import sys

sys.path.append("../")

import time
from networks.mscred.mscred import MSCRED
from common.dataloader import load_dataset
from common.evaluation import evaluate_all
from common.evaluation import compute_salience

# python mscred_benchmark.py --dataset SMAP --lr 0.001 --in_channels_encoder 3 --in_channels_decoder 256 --hidden_size 16 --num_epochs 1 --gpu 2

dataset = "SMD"
subdataset = "machine-1-2"
device = "0"  # cuda:0, a string
step_max = 5
gap_time = 20
win_size = [10, 20, 30]  # sliding window size
in_channels_encoder = 3
in_channels_decoder = 256
save_path = "../mscred_data/" + dataset + "/" + subdataset + "/"
learning_rate = 0.0002
epoch = 1
thred_b = 0.005
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(dataset, subdataset, "all")

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    start = time.time()

    mscred = MSCRED(
        in_channels_encoder,
        in_channels_decoder,
        save_path,
        device,
        step_max,
        gap_time,
        win_size,
        learning_rate,
        epoch,
        thred_b,
    )

    mscred.fit(data_dict)

    end = time.time()

    time = end - start

    anomaly_score, anomaly_label = mscred.predict_prob(
        len(x_train), x_test, x_test_labels
    )

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
    salience = compute_salience(anomaly_score, anomaly_label)
    print('time')
    print('   ', time)
    print('salience')
    print('   ', salience)
