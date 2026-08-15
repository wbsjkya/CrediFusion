import os

import numpy as np
import pandas as pd
import copy
try:
    from sklearn.preprocessing.data import StandardScaler
except:
    from sklearn.preprocessing._data import StandardScaler


def get_top_k_labels(data_dir, top_k=5):
    data_path = "NUS_WIDE/Groundtruth/AllLabels"
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        # print(file)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            # print("label:", label)
            # df = pd.read_csv(os.path.join(data_dir, file))
            df = pd.read_csv(file)
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data_with_2_party(data_dir, selected_labels, n_samples, dtype="Train"):
    # get labels
    data_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_labels:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    if len(selected_labels) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels

    # get XA, which are image low level features
    features_path = "NUS_WIDE/Low_Level_Features"
    if os.path.exists(os.path.join(data_dir, features_path, '{}_XA.pkl'.format(dtype))):
        data_XA = pd.read_pickle(os.path.join(data_dir, features_path, '{}_XA.pkl'.format(dtype)))
    else:
        dfs = []
        for file in os.listdir(os.path.join(data_dir, features_path)):
            if file.startswith("_".join([dtype, "Normalized"])):
                df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
                df.dropna(axis=1, inplace=True)
                print("{0} datasets features {1}".format(file, len(df.columns)))
                dfs.append(df)
        data_XA = pd.concat(dfs, axis=1)
        data_XA.to_pickle(os.path.join(data_dir, features_path, '{}_XA.pkl'.format(dtype)))
    data_XA_selected = data_XA.loc[selected.index]

    # get XB, which are tags
    tag_path = "NUS_WIDE/NUS_WID_Tags/"
    if os.path.exists(os.path.join(data_dir, tag_path, '{}_XB.pkl'.format(dtype))):
        tagsdf = pd.read_pickle(os.path.join(data_dir, tag_path, '{}_XB.pkl'.format(dtype)))
    else:
        file = "_".join([dtype, "Tags1k"]) + ".dat"
        tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
        tagsdf.dropna(axis=1, inplace=True)
        tagsdf.to_pickle(os.path.join(data_dir, tag_path, '{}_XB.pkl'.format(dtype)))
    data_XB_selected = tagsdf.loc[selected.index]
    if n_samples != -1:
        return data_XA_selected.values[:n_samples], data_XB_selected.values[:n_samples], selected.values[:n_samples]
    else:
        # load all data
        return data_XA_selected.values, data_XB_selected.values, selected.values


def load_two_party_data(data_dir, selected_labels, data_type, neg_label=-1, n_samples=-1):
    print("# load_two_party_data")

    Xa, Xb, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                              selected_labels=selected_labels,
                                              n_samples=n_samples, dtype=data_type)

    scale_model = StandardScaler()
    Xa = scale_model.fit_transform(Xa)
    Xb = scale_model.fit_transform(Xb)

    y_ = []
    pos_count = 0
    neg_count = 0
    count = {}
    for i in range(y.shape[0]):
        # the first label in y as the first class while the other labels as the second class
        label = np.nonzero(y[i,:])[0][0]
        y_.append(label)
        if label not in count:
            count[label] = 1
        else:
            count[label] = count[label] + 1
    print(count)

    y = np.expand_dims(y_, axis=1)

    return [Xa, Xb, y]


class NUSWIDEDataset2Party():

    def __init__(self, data_dir, selected_labels, data_type, k=2, neg_label=0, n_samples=-1):
        self.data_dir = data_dir
        self.selected_labels = selected_labels
        self.neg_label = neg_label
        self.n_samples = n_samples
        self.k = k
        [Xa, Xb, y] = load_two_party_data(data_dir, selected_labels, data_type, neg_label, n_samples)
        self.x = [Xa, Xb]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(2):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


class TabularDataTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, percent=0.3, params=None):
        self.percent = percent
        self.params = params

    def __call__(self, x, client_id=0):
        masked_dim = int(x.shape[-1]*self.percent)
        masked_index = np.random.choice(x.shape[-1], masked_dim, replace=False)
        q = copy.deepcopy(x)
        masked_value = np.random.uniform(self.params[client_id][0][masked_index], self.params[client_id][1][masked_index])
        q[masked_index] = masked_value
        return [x, q]


class NUSWIDEAugDataset2Party():

    def __init__(self, data_dir, selected_labels, data_type, k=2, neg_label=0, n_samples=-1):
        self.data_dir = data_dir
        self.selected_labels = selected_labels
        self.neg_label = neg_label
        self.n_samples = n_samples
        self.k = k
        [Xa, Xb, y] = load_two_party_data(data_dir, selected_labels, data_type, neg_label, n_samples)
        print(Xa.shape, Xb.shape)
        self.Xa_min = np.min(Xa, axis=0)
        self.Xa_max = np.max(Xa, axis=0)
        self.Xb_min = np.min(Xb, axis=0)
        self.Xb_max = np.max(Xb, axis=0)
        self.params = [[self.Xa_min, self.Xa_max], [self.Xb_min, self.Xb_max]]

        self.transform = TabularDataTransform(0.3, self.params)

        self.x = [Xa, Xb]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = self.transform(x, i)
            data.append(x)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()

