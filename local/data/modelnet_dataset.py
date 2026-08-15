# encoding: utf-8

import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
import random
import cv2


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

augmentation = [
    transforms.RandomResizedCrop(32, scale=(0.3, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
    ], p=0.8),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class MultiViewAlignedDataset4Party:

    def __init__(self, data_dir, data_type, height, width, k, repeats=1):
        random.seed(0)

        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

        angle = '060'

        self.classes, self.class_to_idx = self.find_class(data_dir)
        subfixes = [str(((i - 1) * 30)).zfill(3) + '_' + str(i).zfill(3) for i in range(1, 13)]

        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            if len(label.split('_')) == 1:
                all_indexes = list(set([item.split('_')[1] for item in all_files]))
            else:
                all_indexes = list(set([item.split('_')[2] for item in all_files]))

            for ind in all_indexes:
                all_views = ['{}_{}_{}_{}.png'.format(label, ind, angle, sg_subfix) for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]

                if data_type == 'train':
                    client_idxs = []
                    for r in range(repeats):
                        for i in range(k):
                            client_idxs.append(list(range(i*3, i*3+3)))
                            random.shuffle(client_idxs[-1])

                    for i in range(len(client_idxs[0])):
                        sample = [all_views[x[i]] for x in client_idxs]
                        self.x.append(sample)
                        self.y.append([self.class_to_idx[label]])

                if data_type == 'test':
                    for i in range(3):
                        sample = [all_views[j*3 + i] for j in range(0, k)]
                        self.x.append(sample)
                        self.y.append([self.class_to_idx[label]])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


"""
local augmentation version
"""


class MultiViewAlignedAugDataset4Party:

    def __init__(self, data_dir, data_type, height, width, k, repeats=1):
        random.seed(0)

        self.x = []
        self.y = []
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = TwoCropsTransform(transforms.Compose(augmentation))

        angle = '060'

        self.classes, self.class_to_idx = self.find_class(data_dir)
        subfixes = [str(((i - 1) * 30)).zfill(3) + '_' + str(i).zfill(3) for i in range(1, 13)]
        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            # all_off_files = ['_'.join(item.split('_')[:-2]) for item in all_files]
            if len(label.split('_')) == 1:
                all_indexes = list(set([item.split('_')[1] for item in all_files]))
            else:
                all_indexes = list(set([item.split('_')[2] for item in all_files]))

            for ind in all_indexes:
                all_views = ['{}_{}_{}_{}.png'.format(label, ind, angle, sg_subfix) for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]

                if data_type == 'train':
                    client_idxs = []
                    for r in range(repeats):
                        for i in range(k):
                            client_idxs.append(list(range(i*3, i*3+3)))
                            random.shuffle(client_idxs[-1])

                    for i in range(len(client_idxs[0])):
                        sample = [all_views[x[i]] for x in client_idxs]
                        self.x.append(sample)
                        self.y.append([self.class_to_idx[label]])

                if data_type == 'test':
                    for i in range(3):
                        sample = [all_views[j*3 + i] for j in range(0, k)]

                        self.x.append(sample)
                        self.y.append([self.class_to_idx[label]])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)

        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()
