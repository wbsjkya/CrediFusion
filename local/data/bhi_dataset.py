# encoding: utf-8

import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
import random
import cv2
from itertools import permutations, combinations
import shutil


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


class BHIDataset2Party:

    def __init__(self, data_dir, data_type, height, width, k, seed=0):
        self.party_num = 2
        self.shuffle_within_patient = False
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
        random.seed(seed)
        patients = [item for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]
        patients_num = len(patients)
        train_num = int(patients_num*0.8)
        random.shuffle(patients)

        if data_type.lower() == 'train':
            patients = patients[:train_num]
        else:
            patients = patients[train_num:]

        for patient in patients:
            for l in [0, 1]:
                files = [d for d in os.listdir(os.path.join(data_dir, patient, str(l)))]
                if self.shuffle_within_patient:
                    random.shuffle(files)
                file_combi_num = len(files) // self.party_num
                for i in range(file_combi_num):
                    sample = [os.path.join(data_dir, patient, str(l), files[2*i+j]) for j in range(self.k)]
                    self.x.append(sample)
                    self.y.append(l)

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


class BHIAugDataset2Party:

    def __init__(self, data_dir, data_type, height, width, k, seed=0):
        self.party_num = 2
        self.shuffle_within_patient = False
        self.x = []
        self.y = []
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = TwoCropsTransform(transforms.Compose(augmentation))

        random.seed(seed)
        patients = [item for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]
        patients_num = len(patients)
        train_num = int(patients_num*0.8)
        random.shuffle(patients)

        if data_type.lower() == 'train':
            patients = patients[:train_num]
        else:
            patients = patients[train_num:]

        for patient in patients:
            for l in [0, 1]:
                files = [d for d in os.listdir(os.path.join(data_dir, patient, str(l)))]
                if self.shuffle_within_patient:
                    random.shuffle(files)
                file_combi_num = len(files) // self.party_num
                for i in range(file_combi_num):
                    sample = [os.path.join(data_dir, patient, str(l), files[2*i+j]) for j in range(self.k)]
                    self.x.append(sample)
                    self.y.append(l)

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