import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing.data import StandardScaler, OneHotEncoder


def series_plot(losses, fscores, aucs):
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['font.serif'] = ['SimHei']
    fig = plt.figure(figsize=(20, 40))

    plt.subplot(311)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.title("loss")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(fscores)
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.title("fscore")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(aucs)
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.title("auc")
    plt.grid(True)

    plt.show()


def balance_X_y(X, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == -1)
    print("pos samples", num_pos)
    print("neg samples", num_neg)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0]

    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        rand_indexes = neg_indexes[:num_pos]
        indexes = pos_indexes + rand_indexes
        y = [y[i] for i in indexes]
        X = [X[i] for i in indexes]
    return np.array(X), np.array(y)


def shuffle_X_y(X, y, seed=5):
    np.random.seed(seed)
    data_size = X.shape[0]
    shuffle_index = list(range(data_size))
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index, :]
    y = y[shuffle_index]
    # print("X,",X)
    return X, y


def convert_to_pos_neg_labels(labels):
    converted_lbls = []
    for lbl in labels:
        if lbl == 0:
            lbl = -1
        converted_lbls.append(lbl)
    return np.array(converted_lbls)


def load_data(infile, balanced=True, seed=5):
    X = []
    y = []
    sids = []

    with open(infile, "r") as fi:
        fi.readline()
        reader = csv.reader(fi)
        for row in reader:
            sids.append(row[0])
            X.append(row[1:-1])
            y0 = int(row[-1])
            if y0 == 0:
                y0 = -1
            y.append(y0)
    y = np.array(y)

    if balanced:
        X, y = balance_X_y(X, y, seed)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    encoder = OneHotEncoder(categorical_features=[1, 2, 3])
    encoder.fit(X)
    X = encoder.transform(X).toarray()

    X, y = shuffle_X_y(X, y, seed)

    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    scale_model = StandardScaler()
    X = scale_model.fit_transform(X)

    return X, np.expand_dims(y, axis=1)


def split_data(X, y, overlap_ratio=0.3, ab_split_ratio=0.2):
    # split datasets
    data_size = X.shape[0]
    feature_a = 11
    overlap_size = int(data_size * overlap_ratio)
    A_size = int((data_size - overlap_size) * (1 - ab_split_ratio))
    B_size = (data_size - overlap_size) - A_size
    X_AB = X[:overlap_size, :]
    y_AB = y[:overlap_size]
    X_A = X[overlap_size:(overlap_size + A_size), 0:feature_a]
    y_A = y[overlap_size:(overlap_size + A_size)]
    X_B = X[overlap_size + A_size:, feature_a:]
    print("X_AB shape:", X_AB.shape)
    print("y_AB shape:", y_AB.shape)
    print("X_A shape:", X_A.shape)
    print("y_A shape:", y_A.shape)
    print("X_B shape:", X_B.shape)
    return X_AB, y_AB, X_A, y_A, X_B


def split_data_combined(X, y, overlap_ratio=0.3, ab_split_ratio=0.1, n_feature_b=16):
    data_size = X.shape[0]
    # n_feature_a = 16
    # n_feature_b = X.shape[1] - n_feature_b
    overlap_size = int(data_size * overlap_ratio)
    overlap_indexes = np.array(range(overlap_size))
    A_size = int((data_size - overlap_size) * (1 - ab_split_ratio))
    # B_size = (data_size - overlap_size) - A_size
    X_A = X[:A_size + overlap_size, n_feature_b:]
    y_A = y[:A_size + overlap_size, :]
    X_B = np.vstack((X[:overlap_size, :n_feature_b], X[A_size + overlap_size:, :n_feature_b]))
    y_B = np.vstack((y[:overlap_size, :], y[A_size + overlap_size:, :]))
    print("X shape:", X.shape)
    print("X_A shape:", X_A.shape)
    print("X_B shape:", X_B.shape)
    print("X_B", X_B)
    print("y_B", y_B)
    print("overlap size:", overlap_size)
    # print(np.sum(y_A[overlap_indexes]>0))
    return X_A, y_A, X_B, y_B, overlap_indexes


def split_data_fixed(X, y, overlap_ratio=0.4, B_size=500, n_feature_b=16):
    data_size = X.shape[0]
    # n_feature_a = 16
    # n_feature_b = X.shape[1] - n_feature_b
    overlap_size = int(data_size * overlap_ratio)
    overlap_indexes = np.array(range(overlap_size))
    # B_size = (data_size-overlap_size) - A_size
    A_size = data_size - overlap_size - B_size  # int((data_size - overlap_size)*(1-ab_split_ratio))

    X_A = X[:A_size + overlap_size, n_feature_b:]
    y_A = y[:A_size + overlap_size, :]
    X_B = np.vstack((X[:overlap_size, :n_feature_b], X[A_size + overlap_size:, :n_feature_b]))
    y_B = np.vstack((y[:overlap_size, :], y[A_size + overlap_size:, :]))
    print("X shape:", X.shape)
    print("X_A shape:", X_A.shape)
    print("X_B shape:", X_B.shape)
    print("overlap size:", overlap_size)
    # print(np.sum(y_A[overlap_indexes]>0))
    return X_A, y_A, X_B, y_B, overlap_indexes


def batch_data_A(X_A, overlap_indexes, batch_size=64, epoches=20):
    n_batches = int(len(overlap_indexes) / batch_size) + 1
    A_non_overlap_indexes = np.setdiff1d(range(X_A.shape[0]), overlap_indexes)
    for i_epoch in range(epoches):
        for i_batch in range(n_batches):
            batch_index_overlap = overlap_indexes[i_batch * batch_size:(i_batch + 1) * batch_size]
            batch_index_nonoverlap = A_non_overlap_indexes[i_batch * batch_size:(i_batch + 1) * batch_size]
            yield (batch_index_overlap, batch_index_nonoverlap)


def batch_data_B(overlap_indexes, batch_size=64, epoches=20):
    n_batches = int(len(overlap_indexes) / batch_size) + 1
    for i_epoch in range(epoches):
        for i_batch in range(n_batches):
            batch_index_overlap = overlap_indexes[i_batch * batch_size:(i_batch + 1) * batch_size]
            yield (batch_index_overlap)


def split_data_all(X, y, overlap_ratio=0.3, ab_split_ratio=0.2):
    # split datasets
    data_size = X.shape[0]
    feature_a = 11
    overlap_size = int(data_size * overlap_ratio)
    A_size = int((data_size - overlap_size) * (1 - ab_split_ratio))
    B_size = (data_size - overlap_size) - A_size

    X_A = X[:(overlap_size + A_size)]
    X_A_in_common = np.zeros(shape=(overlap_size + A_size, 2))
    X_A_in_common[:overlap_size, 0] = 1
    X_A_in_common[overlap_size:, 1] = 1
    y_A = y[:(overlap_size + A_size)]
    X_B = X[overlap_size + A_size:]
    Y_B = y[overlap_size + A_size:]

    print("X_A shape:", X_A.shape)
    print("y_A shape:", y_A.shape)
    print("X_B shape:", X_B.shape)
    return X_A, X_A_in_common, y_A, X_B, Y_B


def get_timestamp():
    local_time = time.localtime(time.time())
    timestamp = time.strftime("%Y%m%d%H%M%S", local_time)
    return timestamp


def save_result(*, file_full_name, loss_records, metric_one_records, metric_two_records, spend_time_records=None):
    file_full_name = file_full_name + ".csv"
    print("save result to {0}".format(file_full_name))
    with open(file_full_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for loss_list in loss_records:
            wr.writerow(loss_list)
        for metric_one_list in metric_one_records:
            wr.writerow(metric_one_list)
        for metric_two_list in metric_two_records:
            wr.writerow(metric_two_list)

        if spend_time_records is not None:
            for spend_time_list in spend_time_records:
                wr.writerow(spend_time_list)


def plot_result(lengend_list, loss_records, metric_test_records, metric_train_records, metric_name, if_metric_log):
    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    # style_list = ["r", "g", "g--", "k", "k--", "y", "y--"]
    # style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]

    style_list = ["r", "b", "g", "k", "m", "y", "c"]

    if len(lengend_list) == 6:
        style_list = ["r", "b", "g", "k", "m", "y", "c"]

    if len(lengend_list) == 9:
        style_list = ["r", "r--", "r:", "b", "b--", "b:", "g", "g--", "g:"]

    markevery= 50
    markesize = 3

    plt.subplot(131)
    for i, loss_list in enumerate(loss_records):
        if if_metric_log[0]:
            plt.semilogy(loss_list, style_list[i], markersize=markesize, markevery=markevery)
        else:
            plt.plot(loss_list, style_list[i], markersize=markesize, markevery=markevery)
            plt.ylim(0.6, 1.0)
    plt.xlabel("communication rounds")
    plt.ylabel("loss")
    plt.legend(lengend_list, loc='best')

    plt.subplot(132)
    for i, metric_test_list in enumerate(metric_test_records):
        if if_metric_log[1]:
            plt.semilogy(metric_test_list, style_list[i], markersize=markesize, markevery=markevery)
        else:
            plt.plot(metric_test_list, style_list[i], markersize=markesize, markevery=markevery)
    plt.xlabel("communication rounds")
    plt.ylabel("test " + metric_name)
    plt.legend(lengend_list, loc='best')

    plt.subplot(133)
    for i, metric_train_list in enumerate(metric_train_records):
        if if_metric_log[2]:
            plt.semilogy(metric_train_list, style_list[i], markersize=markesize, markevery=markevery)
        else:
            plt.plot(metric_train_list, style_list[i], markersize=markesize, markevery=markevery)
    plt.xlabel("communication rounds")
    plt.ylabel("train " + metric_name)
    plt.legend(lengend_list, loc='best')

    # file_full_name_eps = file_full_name + ".eps"
    # file_full_name_png = file_full_name + ".png"
    # plt.savefig(file_full_name_eps, format='eps')
    # plt.savefig(file_full_name_png, format='png')
    plt.show()


def compute_experimental_result_file_name(n_local, batch_size, comm_rounds):
    local_epochs = "L_" + str(n_local)
    batch_size = "B_" + str(batch_size)
    comm_rounds = "R_" + str(comm_rounds)
    return local_epochs + "_" + batch_size + "_" + comm_rounds


if __name__ == '__main__':
    # X,y = load_data()
    # load_data()
    a = np.setdiff1d([1, 2, 3, 2], [2])
    print(a)
