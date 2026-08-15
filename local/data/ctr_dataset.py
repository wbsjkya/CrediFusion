# encoding: utf-8

import os
import platform
import numpy as np
import pandas as pd
import torch
import copy
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from models.ctr_model_utils import build_input_features, SparseFeat, DenseFeat, get_feature_names


class Avazu2party:

    def __init__(self, data_dir, data_type, k, input_size):
        self.x = []
        self.y = []
        self.data_dir = data_dir
        self.k = k
        self.data_dir = data_dir

        # split features
        self.feature_list = []

        sparse_features = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                            'device_id', 'device_ip', 'device_model','C14','C17','C19','C20','C21']
        dense_features = ['C1', 'banner_pos', 'device_type', 'device_conn_type', 'C15','C16', 'C18']

        sparse_features_list = [sparse_features[7:], sparse_features[:7]]
        dense_features_list = [dense_features[4:], dense_features[:4]]

        # train = pd.read_csv(os.path.join(self.data_dir, 'train_10W.txt'))  # , sep='\t', header=None)
        # train = pd.read_csv(os.path.join(self.data_dir, 'Train'))  # , sep='\t', header=None)
        # test = pd.read_csv(os.path.join(self.data_dir, 'Test'))  # , sep='\t', header=None)
        # # test = pd.read_csv(os.path.join(self.data_dir, 'test_2W.txt'))  # , sep='\t', header=None)
        # data = pd.concat([train, test], axis=0)
        data = pd.read_csv(os.path.join(self.data_dir, 'train_subset.csv'))  # , sep='\t', header=None)

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        if platform.system() == 'Windows':
             train, test = train_test_split(data, test_size=0.2, shuffle=False)
        else:
            train = data.iloc[:800000]
            test = data.iloc[800000:]

        if data_type.lower() == 'train':
            labels = train['click']
        else:
            labels = test['click']
        self.y = labels.values

        self.x = []
        self.feature_dim = []
        # 2.count #unique features for each sparse field,and record dense feature field name
        for i in range(len(sparse_features_list)):
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), input_size)
                                      for feat in sparse_features_list[i]] + [DenseFeat(feat, 1, )
                                                                      for feat in dense_features_list[i]]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            self.feature_list.append(fixlen_feature_columns)

            feature_names = get_feature_names(
                linear_feature_columns + dnn_feature_columns)

            # 3.generate input data for model
            if data_type.lower() == 'train':
                x = {name: train[name] for name in feature_names}
            else:
                x = {name: test[name] for name in feature_names}
            feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)

            if isinstance(x, dict):
                x = [x[feature] for feature in feature_index]

            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            x = np.concatenate(x, axis=-1)
            self.x.append(x)
        del data, train, test

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, indexx):  # this is single_indexx
        labels = []
        data = [self.x[0][indexx], self.x[1][indexx]]
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()


class CtrDataTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, percent=0.3, col_infos=None):
        self.percent = percent
        self.col_infos = col_infos
        self.masked_sparse_dim_list = [int(round(len(self.col_infos[i]['sparse_idx'])*self.percent)) for i in range(2)]
        self.masked_dense_dim_list = [int(round(len(self.col_infos[i]['dense_idx'])*self.percent)) for i in range(2)]

        self.sparse_idx_list = [self.col_infos[i]['sparse_idx'] for i in range(2)]
        self.dense_idx_list = [self.col_infos[i]['dense_idx'] for i in range(2)]
        self.sparse_voc_list = [np.array(self.col_infos[i]['sparse_voc']) for i in range(2)]

    def __call__(self, x, client_id=0):
        masked_sparse_dim = self.masked_sparse_dim_list[client_id]
        masked_dense_dim = self.masked_dense_dim_list[client_id]

        q = copy.deepcopy(x)

        if masked_sparse_dim > 0:
            masked_sparse_index = np.random.choice(self.sparse_idx_list[client_id], masked_sparse_dim,  replace=False)
            masked_sparse_value = [v - 1 for v in self.sparse_voc_list[client_id][masked_sparse_index]]
            q[masked_sparse_index] = masked_sparse_value

        if masked_dense_dim > 0:
            masked_dense_index = np.random.choice(self.dense_idx_list[client_id], masked_dense_dim,  replace=False)

            masked_dense_value = np.random.uniform(0.0, 1.0, len(masked_dense_index))
            q[masked_dense_index] = masked_dense_value

        return [x, q]


class AvazuAug2party:

    def __init__(self, data_dir, data_type, k, input_size):
        self.x = []
        self.y = []
        self.data_dir = data_dir
        self.k = k
        train_ratio = 0.2
        self.data_dir = data_dir

        # split features
        self.feature_list = []

        sparse_features = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                            'device_id', 'device_ip', 'device_model','C14','C17','C19','C20','C21']
        dense_features = ['C1', 'banner_pos', 'device_type', 'device_conn_type', 'C15','C16', 'C18']

        sparse_features_list = [sparse_features[7:], sparse_features[:7]]
        dense_features_list = [dense_features[4:], dense_features[:4]]
        # train = pd.read_csv(os.path.join(self.data_dir, 'Train'))  # , sep='\t', header=None)
        # # train = pd.read_csv(os.path.join(self.data_dir, 'train_10W.txt'))  # , sep='\t', header=None)
        # test = pd.read_csv(os.path.join(self.data_dir, 'Test'))  # , sep='\t', header=None)
        # # test = pd.read_csv(os.path.join(self.data_dir, 'test_2W.txt'))  # , sep='\t', header=None)
        # data = pd.concat([train, test], axis=0)
        data = pd.read_csv(os.path.join(self.data_dir, 'train_subset.csv'))  # , sep='\t', header=None)

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        train = data.iloc[:800000]
        test = data.iloc[800000:]

        if data_type.lower() == 'train':
            labels = train['click']
        else:
            labels = test['click']
        self.y = labels.values

        self.x = []
        self.feature_dim = []
        self.col_info_list = []

        # 2.count #unique features for each sparse field,and record dense feature field name
        for i in range(len(sparse_features_list)):
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()+1, input_size)
                                      for feat in sparse_features_list[i]] + [DenseFeat(feat, 1, )
                                                                      for feat in dense_features_list[i]]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            self.feature_list.append(fixlen_feature_columns)

            feature_names = get_feature_names(
                linear_feature_columns + dnn_feature_columns)

            # 3.generate input data for model
            if data_type.lower() == 'train':
                x = {name: train[name] for name in feature_names}
            else:
                x = {name: test[name] for name in feature_names}

            feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)

            sparse_list = list(
                filter(lambda x: isinstance(x, SparseFeat), fixlen_feature_columns)) if len(
                fixlen_feature_columns) else []
            dense_list = list(
                filter(lambda x: isinstance(x, DenseFeat), fixlen_feature_columns)) if len(
                fixlen_feature_columns) else []
            sparse_idx = [feature_index[feat.name][0] for feat in sparse_list]
            dense_idx = [feature_index[feat.name][0] for feat in dense_list]
            sparse_voc_size = [feat.vocabulary_size for feat in sparse_list]
            self.col_info_list.append({'sparse_idx': sparse_idx, 'sparse_voc': sparse_voc_size, 'dense_idx': dense_idx})

            if isinstance(x, dict):
                x = [x[feature] for feature in feature_index]

            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            x = np.concatenate(x, axis=-1)
            self.x.append(x)
        self.transform = CtrDataTransform(0.3, self.col_info_list)

        del data, train, test

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, indexx):  # this is single_indexx
        labels = []
        data = []

        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = self.transform(x, i)
            data.append(x)

        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


