import argparse
import logging
import os
import random
import sys
import time
import numpy as np

import torch.backends.cudnn as cudnn
import torch.utils
from tensorboardX import SummaryWriter

import utils

# dataset
from data.bhi_dataset import BHIDataset2Party, BHIAugDataset2Party
from data.ctr_dataset import Avazu2party, AvazuAug2party
from data.modelnet_dataset import MultiViewAlignedDataset4Party, MultiViewAlignedAugDataset4Party
from data.nuswide_dataset_multi import NUSWIDEDataset2Party, NUSWIDEAugDataset2Party

# model
from models.model_templates import *
from models.ctr_model import *


def prepare_exp(exp_type='pretrain'):
    parser = argparse.ArgumentParser("main_ext")

    # general
    parser.add_argument('--dataset', type=str, default='nuswide10classes2party',
                        help='dataset')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    parser.add_argument('--experiment_dir', default='experiment', help='experiment save dir')
    parser.add_argument('--time_in_name', type=int, default=1, help='epochs between two learning rate decays')

    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')

    # dataset
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--input_size', type=int, default=32, help='resnet')
    parser.add_argument('--client_idx', type=str, default='')
    parser.add_argument('--label_percent', type=float, default=1.0, help='')
    parser.add_argument('--train_used_num', type=int, default=200, help='num of fine-tuning sample')
    parser.add_argument('--aligned_label_percent', type=float, default=0.4, help='aligned sample percent')
    parser.add_argument('--valid_percent', type=float, default=0.0)

    # model
    parser.add_argument('--model', default='mlp2', help='resnet')
    parser.add_argument('--num_cls_layer', type=int, default=1, help='layers of the classification head')

    # training
    parser.add_argument('--exp_type', type=str, default='pretrain', help='gradient clipping for weights')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd')

    # training: pretrain
    parser.add_argument('--local_ssl', type=int, default=1, help='0: disable local ssl; 1: enable local ssl')
    parser.add_argument('--aggregation_mode', type=str, default='none', help='none: no agg; pma: partial agg')

    parser.add_argument('--pretrain_method', type=str, default='simsiam')
    parser.add_argument('--local_epochs_cross', type=int, default=1)
    parser.add_argument('--local_epochs_local', type=int, default=1)
    # parser.add_argument('--comm_mode', type=str, default='first', help='[first, all]')
    parser.add_argument('--comm_mode', type=str, default=None, help='[first, all]')
    parser.add_argument('--pretrain_lr_encoder', type=float, default=0.05)
    parser.add_argument('--pretrain_lr_head', type=float, default=0.05)
    parser.add_argument('--pretrain_model_dir', default='premodels', help='save dir for pretrained model')

    parser.add_argument('--pretrain_lr_decay', type=int, default=1, help='0: constant; 1: cosine decay ,except for predictor')
    parser.add_argument('--constraint_ratio', type=float, default=0.5, help='constraint on local model output')
    # parser.add_argument('--use_sim', type=bool, default=True, help='use instance similarity for local')
    parser.add_argument('--miu', type=float, default=0.5, help=' ')
    parser.add_argument('--constraint_pre_ratio', type=float, default=0.0, help='constraint on local model output')
    parser.add_argument('--use_FISL', type=int, default=1, help='whether to use FISL')
    parser.add_argument('--dis_power', type=float, default=2.0, help='constraint on distillation')
    parser.add_argument('--temp', type=float, default=0.03, help='temp on distillation')

    parser.add_argument('--local_ratio', type=float, default=0.5, help='learning rate for ')
    parser.add_argument('--pt_feat_iso_sigma', type=float, default=0.0, help='defense strength of feature in pretraining phase')
    parser.add_argument('--pt_model_iso_sigma', type=float, default=0.0, help='defense strength of model in pretraining phase')
    parser.add_argument('--pt_iso_threshold', type=float, default=5.0, help='clamp threshold')



    # training: pretrain head dimension
    parser.add_argument('--out_dim', type=int, default=512, help='out dim of head')
    parser.add_argument('--proj_hidden_dim', type=int, default=512, help='proj_hidden_dim')
    parser.add_argument('--pred_hidden_dim', type=int, default=128, help='pred_hidden_dim')
    parser.add_argument('--num_ftrs', type=int, default=512, help='feature dimension')
    parser.add_argument('--proj_layer', type=int, default=3, help='projector layer')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of mlp encoder')
    parser.add_argument('--pool', type=str, default='mean', help='pooling method for vfl classification task')

    # training: cls
    parser.add_argument('--pretrained_path', type=str, default='', help='gradient clipping for weights')
    parser.add_argument('--freeze_backbone', type=int, default=0, help='0: no freeze; 1: freeze all; 2: freeze passive')
    parser.add_argument('--use_local_model', type=int, default=1, help='whether to use local model')
    parser.add_argument('--use_cross_model', type=int, default=1, help='whether to use local model')
    parser.add_argument('--cls_iso_sigma', type=float, default=0.0, help='coef for mutual training from local to cross')
    parser.add_argument('--cls_iso_threshold', type=float, default=5.0, help='constraint on local model output')
    parser.add_argument('--cls_model_dir', default='clsmodels', help='clsmodels')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')

    args = parser.parse_args()
    args.exp_type = exp_type

    if args.client_idx == '':
        args.client_idx = list(range(0, args.k))
    else:
        args.client_idx = eval(args.client_idx)

    args.name = '{}/{}-{}'.format(args.experiment_dir, args.name, time.strftime("%Y%m%d-%H%M%S"))

    utils.create_exp_dir(args.name, scripts_to_save=None)

    # set up logger
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
    writer.add_text('experiment', args.name, 0)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    logging.info('***** USED DEVICE: {}'.format(device))
    logging.info('***** client idx： {}'.format(args.client_idx))

    assert len(args.client_idx) == args.k, print('incompatible clients number and client index')
    return args, writer


def save_models_from_clients(client_list, args, epochs=None):
    # define name_str as desired
    name_str = args.model
    for i in range(args.k):
        client_list[i].save_models(args.pretrain_model_dir, name_str, i)


def save_models_from_passive_client(cls_model, args):
    # used for label leakage attack, save the pretrained models or finetuned models
    # define name_str as desired
    name_str = args.model
    cls_model.save_models(args.cls_model_dir, name_str)


def get_dataset(args):
    input_dims = None
    train_dataset = None
    test_dataset = None

    if args.dataset == 'mn4party':
        NUM_CLASSES = 40
        DATA_DIR = './../../dataset/modelnet_aligned/'
        if args.exp_type == 'cls':
            train_dataset = MultiViewAlignedDataset4Party(DATA_DIR, 'train', args.input_size, args.input_size, 4)
            test_dataset = MultiViewAlignedDataset4Party(DATA_DIR, 'test', args.input_size, args.input_size, 4)
        else:
            train_dataset = MultiViewAlignedDataset4Party(DATA_DIR, 'train', args.input_size, args.input_size, 4)
            train_dataset_aug = MultiViewAlignedAugDataset4Party(DATA_DIR, 'train', args.input_size,
                                                                     args.input_size, 4)
            test_dataset = MultiViewAlignedDataset4Party(DATA_DIR, 'test', args.input_size, args.input_size, 4)
            test_dataset_aug = MultiViewAlignedAugDataset4Party(DATA_DIR, 'test', args.input_size, args.input_size, 4)

    if args.dataset == 'nuswide10classes2party':
        input_dims = [634, 1000]
        if args.k > 2:
            args.k = 2
        sel_lbls = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
        NUM_CLASSES = len(sel_lbls)
        DATA_DIR = './dataset/'
        if args.exp_type == 'cls':
            train_dataset = NUSWIDEDataset2Party(DATA_DIR, sel_lbls, 'Train', 2)
            test_dataset = NUSWIDEDataset2Party(DATA_DIR, sel_lbls, 'Test', 2)
        else:
            train_dataset = NUSWIDEDataset2Party(DATA_DIR, sel_lbls, 'Train', 2)
            train_dataset_aug = NUSWIDEAugDataset2Party(DATA_DIR, sel_lbls, 'Train', 2)
            test_dataset = NUSWIDEDataset2Party(DATA_DIR, sel_lbls, 'Test', 2)
            test_dataset_aug = NUSWIDEAugDataset2Party(DATA_DIR, sel_lbls, 'Test', 2)

    if args.dataset == 'ctr_avazu2party':
        input_dims = [10, 11]
        if args.k > 2:
            args.k = 2
        NUM_CLASSES = 1
        DATA_DIR = './dataset/avazu'
        if args.exp_type == 'cls':
            train_dataset = Avazu2party(DATA_DIR, 'Train', 2, args.input_size)
            test_dataset = Avazu2party(DATA_DIR, 'Test', 2, args.input_size)
            train_dataset_aug = AvazuAug2party(DATA_DIR, 'Train', 2, args.input_size)
        else:
            train_dataset = Avazu2party(DATA_DIR, 'Train', 2, args.input_size)
            train_dataset_aug = AvazuAug2party(DATA_DIR, 'Train', 2, args.input_size)
            test_dataset = Avazu2party(DATA_DIR, 'Test', 2, args.input_size)
            test_dataset_aug = AvazuAug2party(DATA_DIR, 'Test', 2, args.input_size)
    if args.dataset == 'bhi2party':
        args.k = 2 if args.k > 3 else args.k
        NUM_CLASSES = 1
        DATA_DIR = './dataset/bhi'
        if args.exp_type == 'cls':
            train_dataset = BHIDataset2Party(DATA_DIR, 'train', args.input_size, args.input_size, 2)
            test_dataset = BHIDataset2Party(DATA_DIR, 'test', args.input_size, args.input_size, 2)
        else:
            train_dataset = BHIDataset2Party(DATA_DIR, 'train', args.input_size, args.input_size, 2)
            train_dataset_aug = BHIAugDataset2Party(DATA_DIR, 'train', args.input_size, args.input_size, 2)
            test_dataset = BHIDataset2Party(DATA_DIR, 'test', args.input_size, args.input_size, 2)
            test_dataset_aug = BHIAugDataset2Party(DATA_DIR, 'test', args.input_size, args.input_size, 2)

    args.num_classes = NUM_CLASSES
    args.input_dims = input_dims
    if 'ctr' in args.dataset:
        args.col_names = train_dataset_aug.feature_list

    assert train_dataset is not None, print('invalid dataset name')
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    train_indices = list(range(n_train))
    test_indices = list(range(n_test))

    random.shuffle(train_indices)

    logging.info("***** train/valid data num： {}, {}".format(len(train_indices), len(test_indices)))

    train_loader_aligned = None
    train_loader_local = None
    valid_loader = None
    test_loader = None

    if args.exp_type == 'pretrain':
        # aligned samples
        aligned_num = int(n_train * args.aligned_label_percent)

        valid_num_aligned = int(aligned_num * args.valid_percent)
        train_num_aligned = aligned_num - valid_num_aligned
        train_indices_aligned = train_indices[:train_num_aligned]

        valid_num_local = int(n_train * args.valid_percent)
        train_num_local = n_train - valid_num_local
        train_indices_local = train_indices[:train_num_local]

        logging.info("***** train_num_aligned:{}; valid_num_aligned:{}".format(train_num_aligned, valid_num_aligned))
        logging.info("***** train_num_local:{}; valid_num_local:{}".format(train_num_local, valid_num_local))

        train_sampler_aligned = torch.utils.data.sampler.SubsetRandomSampler(train_indices_aligned)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        train_loader_aligned = get_loader(train_dataset, train_sampler_aligned, args)

        test_loader = [get_loader(test_dataset, test_sampler, args), get_loader(test_dataset_aug, test_sampler, args)]

        # if use valid data
        if args.valid_percent > 0:
            valid_indices_aligned = train_indices[train_num_aligned:aligned_num]
            valid_indices_local = train_indices[train_num_local:]
            valid_sampler_aligned = torch.utils.data.sampler.SubsetRandomSampler(valid_indices_aligned)
            valid_sampler_local = torch.utils.data.sampler.SubsetRandomSampler(valid_indices_local)

            valid_loader_aligned = get_loader(train_dataset, valid_sampler_aligned, args)
            valid_loader_local = get_loader(train_dataset, valid_sampler_local, args)
            valid_loader = [valid_loader_aligned, valid_loader_local]
        # local ssl
        if args.local_ssl:
            train_sampler_local = torch.utils.data.sampler.SubsetRandomSampler(train_indices_local)
            train_loader_local = get_loader(train_dataset_aug, train_sampler_local, args)

    elif args.exp_type == 'cls':
        aligned_num = int(n_train * args.aligned_label_percent)
        if args.label_percent > 1:
            train_used_num = int(args.label_percent)
            assert train_used_num < len(train_indices), print('train sample number exceeds maximal sample num')
        else:
            # train_used_num = int(n_train * args.aligned_label_percent * args.label_percent)
            train_used_num = int(args.train_used_num)
        train_indices_used = train_indices[:train_used_num]

        logging.info("***** Used train data {}, aligned data {}, ratio： {}".format(len(train_indices_used), aligned_num,
                                                                                   len(train_indices_used) / len(
                                                                                       train_indices)))

        train_sampler_used = torch.utils.data.sampler.SubsetRandomSampler(train_indices_used)

        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
        train_loader_aligned = get_loader(train_dataset, train_sampler_used, args)

        test_loader = get_loader(test_dataset, valid_sampler, args)

        train_loader_local = None
        valid_loader = None

    assert train_loader_aligned is not None or test_loader is not None, print('invalid dataloader')

    return train_loader_aligned, train_loader_local, valid_loader, test_loader, args


def get_loader(dataset, sampler, args):
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                                       pin_memory=False, drop_last=False)


def get_model(args):
    encoder_models_local_bottom = []
    encoder_models_local_top = []
    encoder_models_cross = []

    if args.model == 'resnet':
        logging.info('***** USE RESNET18 *****')
        for i in range(args.k):
            encoder_model_local_bottom = BottomResnet18()
            encoder_models_local_bottom.append(encoder_model_local_bottom)
            encoder_model = TopResnet18(args.num_classes, output_dim=args.num_ftrs)
            encoder_models_local_top.append(nn.Sequential(*list(encoder_model.children())[:-1]))
            encoder_model = MyResnet18(class_num=10, output_dim=args.num_ftrs)
            encoder_models_cross.append(nn.Sequential(*list(encoder_model.children())[:-1]))
    elif args.model == 'mlp2':
        num_ftrs = args.num_ftrs
        hidden_dim = args.hidden_dim
        for i in range(args.k):
            encoder_model_local_bottom = BottomMLP2(args.input_dims[args.client_idx[i]],hidden_dim)
            encoder_models_local_bottom.append(encoder_model_local_bottom)
            encoder_model = TopMLP2([hidden_dim, num_ftrs])
            encoder_models_local_top.append(encoder_model)
            encoder_model = MLP2(args.input_dims[args.client_idx[i]], [hidden_dim, num_ftrs])
            encoder_models_cross.append(encoder_model)
    elif args.model == 'dnnfm':
        hidden_dim = args.hidden_dim
        num_ftrs = args.num_ftrs
        for i in range(args.k):
            encoder_model_local_bottom = BottomDNNFM(args.col_names[args.client_idx[i]],
                                              args.col_names[args.client_idx[i]], dnn_hidden_units=[hidden_dim])
            encoder_models_local_bottom.append(encoder_model_local_bottom)
            encoder_model = TopDNNFM(hidden_dims=[hidden_dim, num_ftrs])
            encoder_models_local_top.append(encoder_model)
            encoder_model = DNNFM(args.col_names[args.client_idx[i]], args.col_names[args.client_idx[i]],
                                  dnn_hidden_units=[hidden_dim, num_ftrs])
            encoder_models_cross.append(encoder_model)

    return encoder_models_local_bottom, encoder_models_local_top, encoder_models_cross, args


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args, _ = prepare_exp()