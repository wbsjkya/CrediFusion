import copy
import logging
import math
from datetime import datetime

import torch
import torch.utils
from collections import OrderedDict

# import pretrain_simsiam
# import pretrain_byol
import pretrain_moco

from pretrain_template import get_server
import utils
from prepare_experiments import get_dataset, get_model, prepare_exp, set_random_seed, save_models_from_clients
import torch.nn.functional as F


def main():
    """ load args """
    args, tb_writer = prepare_exp('pretrain')

    """ set seed """
    set_random_seed(args.seed)

    """ set gpu """
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logging.info('gpu device = %d' % args.gpu)

    """ get dataloader """
    train_loader_aligned, train_loader_local, valid_loader, test_loader, args = get_dataset(args)

    """ get model """
    encoder_models_local_bottom, encoder_models_local_top, encoder_models_cross, args = get_model(args)

    """ save args to log """
    if 'ctr' in args.dataset:
        args_log = copy.deepcopy(args)
        args_log.col_names = ''
        logging.info("args = {}".format(vars(args_log)))
    else:
        logging.info("args = {}".format(vars(args)))

    """ get method """
    if args.pretrain_method == 'simsiam':
        pretrain_func = pretrain_simsiam
    elif args.pretrain_method == 'byol':
        pretrain_func = pretrain_byol
    elif args.pretrain_method == 'moco':
        pretrain_func = pretrain_moco
    else:
        raise Exception("pretrain method does not support : {}".format(args.pretrain_method))

    """ get clients """
    client_list = pretrain_func.get_clients(encoder_models_local_bottom, encoder_models_local_top, encoder_models_cross,
                                            args)
    logging.info('client generated: {}'.format(len(client_list)))

    """ get server for aggregation """
    Server = get_server([], None, args)

    """ start training """
    best_valid_loss = None
    best_epoch = 0

    # TODO add load weight for clients, from Bottom
    # for i, client in enumerate(client_list):
    #     if i == 0:
    #         state_dict = torch.load('guest-rank(0).pt')
    #     elif i >= 1:
    #         state_dict = torch.load('host-rank({i}).pt')
        
    #     # [state_dict.pop(k) for k in [k for k in state_dict.keys() if (k.startswith('_top') or k.startswith('_agg_layer'))]]
    #     new_state_dict_encoder_cross = {}
    #     new_state_dict_projection_mlp_cross = {}
    #     for key, value in state_dict.items():
    #         if key.startswith("_bottom_model.0"):
    #             new_key = key.replace("_bottom_model.0.", "")
    #             new_state_dict_encoder_cross[new_key] = value
    #         elif key.startswith("_bottom_model.1"):
    #             new_key = key.replace("_bottom_model.1.", "")
    #             new_state_dict_projection_mlp_cross[new_key] = value

    #     client.encoder_cross.load_state_dict(new_state_dict_encoder_cross)
    #     # MLP2(args.input_dims[args.client_idx[i]], [hidden_dim, num_ftrs])
    #     # MLP2([634, 1000][args.client_idx[i]], [512, 512])
    #     # dict_keys(['layer1.0.weight', 'layer1.0.bias', 'layer1.1.weight', 'layer1.1.bias', 'layer1.1.running_mean', 'layer1.1.running_var', 'layer1.1.num_batches_tracked', 'layer2.0.weight', 'layer2.0.bias', 'layer2.1.weight', 'layer2.1.bias', 'layer2.1.running_mean', 'layer2.1.running_var', 'layer2.1.num_batches_tracked'])
    #     client.projection_mlp_cross.load_state_dict(new_state_dict_projection_mlp_cross)
    #     # projection_mlp(args.num_ftrs, self.proj_hidden_dim, self.out_dim, self.num_mlp_layers)
    #     # projection_mlp(512, 512, 512, 3)
    #     # dict_keys(['0.0.weight', '0.1.weight', '0.1.bias', '0.1.running_mean', '0.1.running_var', '0.1.num_batches_tracked', '1.0.weight', '1.1.weight', '1.1.bias', '1.1.running_mean', '1.1.running_var', '1.1.num_batches_tracked', '2.0.weight', '2.0.bias', '2.1.weight', '2.1.bias', '2.1.running_mean', '2.1.running_var', '2.1.num_batches_tracked'])

    for epoch in range(args.epochs):
        # ----- train ----- #
        # assume the first party holds labels
        # 'first' mode means party 1 sends its z_1 to the rest and other parties only sends their z_i to party 1
        # other comm_mode not implemented, e.g. 'all'
        # if args.comm_mode == 'first':
        #     train_cross_model(train_loader_aligned, client_list, epoch, args, tb_writer)

        if args.local_ssl != 0:
            train_local_model(train_loader_local, client_list, Server, epoch, args, tb_writer)
        if args.pretrain_lr_decay == 1:
            for client in client_list:
                client.adjust_learning_rate()
            Server.adjust_learning_rate()

        # ----- validation ----- #
        if args.valid_percent != 0:
            loss_cross, loss_local = valid(valid_loader, client_list, epoch, args, tb_writer)
        else:
            loss_cross, loss_local = valid(test_loader, client_list, epoch, args, tb_writer)

        # save best model
        if best_valid_loss is None:
            best_valid_loss = loss_cross
        else:
            if loss_cross < best_valid_loss:
                best_valid_loss = loss_cross
                best_epoch = epoch
                save_models_from_clients(client_list, args, epoch)
                logging.info("***** best model saved at epoch {} *****".format(epoch))
        logging.info("***** best loss {}: {} *****".format(best_epoch, best_valid_loss))

    """ postprocess training """
    # ----- save pretrained models ----- #
    save_models_from_clients(client_list, args)
    logging.info("***** model saved *****")

    # ----- clean up ----- #
    logging.info("***** results logged *****")
    tb_writer.close()


def train_cross_model(train_loader_aligned, client_list, epoch, args, writer):
    """ main function for cross-party SSL """
    sample_num = len(train_loader_aligned)
    cur_lr = client_list[0].optimizer_list_cross[0].param_groups[0]['lr']
    logging.info("Cross-Party Train Epoch {}, training on aligned data, LR: {}, sample: {}".format(epoch, cur_lr,
                                                                                       sample_num * args.batch_size))
    writer.add_scalar('train_aligned/lr', cur_lr, epoch)

    for client in client_list:
        client.set_train()

    loss, meta = step_cross_model(train_loader_aligned, client_list, epoch, args, 'train')
    loss_per_client = meta['loss_per_client']

    logging.info("Cross-Party SSL Train Epoch {}, client loss aligned: {}".format(epoch, loss_per_client))

    writer.add_scalar('train/loss_aligned', loss, epoch)
    for i, item in enumerate(loss_per_client):
        writer.add_scalar('train/loss_aligned_{}'.format(i), item, epoch)


def train_local_model(train_loader_local, client_list, Server, epoch, args, writer):
    """ main function for guided local SSL """
    sample_num = len(train_loader_local)
    try:
        cur_lr = client_list[0].optimizer_list_local[0].param_groups[0]['lr']
    except:
        cur_lr = client_list[0].optimizer_list_cross[0].param_groups[0]['lr']

    logging.info(
        "Local SSL Train Epoch {}, training on local data, sample: {}".format(epoch, sample_num * args.batch_size))
    writer.add_scalar('train_local/lr', cur_lr, epoch)

    for client in client_list:
        client.set_train()

    loss, meta = step_local_model(train_loader_local, client_list, epoch, args, 'train')
    loss_per_client = meta['loss_per_client']

    logging.info("Local SSL Train Epoch {}, client loss local: {}".format(epoch, loss_per_client))

    writer.add_scalar('train/loss_local', loss, epoch)
    for i, item in enumerate(loss_per_client):
        writer.add_scalar('train/loss_local_{}'.format(i), item, epoch)

    # server: aggregation , local: replace
    loss_agg = Server.aggregation(client_list, sample_num)
    logging.info("Local SSL Train Epoch {}, AGG MODE {}, client loss agg: {}".format(epoch, args.aggregation_mode,
                                                                                     loss_agg))


def valid(valid_loader, client_list, epoch, args, writer):
    """ validation function """
    for client in client_list:
        client.set_eval()

    loss_cross = 0
    loss_local = 0
    with torch.no_grad():
        # same valid loader for both cross-party and local SSL
        if not isinstance(valid_loader, list):
            loss_cross, meta_cross = step_cross_model(valid_loader, client_list, epoch, args, 'valid')
            if args.local_ssl != 0:
                loss_local, meta_local = step_local_model(valid_loader, client_list, epoch, args, 'valid')
        else:
            loss_cross, meta_cross = step_cross_model(valid_loader[0], client_list, epoch, args, 'valid')
            if args.local_ssl != 0:
                loss_local, meta_local = step_local_model(valid_loader[1], client_list, epoch, args, 'valid')

    logging.info("###### Valid Epoch {} Start #####".format(epoch))
    logging.info("Valid Epoch {}, valid client loss aligned: {}".format(epoch, meta_cross['loss_per_client']))

    if args.local_ssl != 0:
        logging.info("Valid Epoch {}, valid client loss local: {}".format(epoch, meta_local['loss_per_client']))
        logging.info("Valid Epoch {}, valid client loss regularized: {}".format(epoch, meta_local['loss_per_client_reg']))

    logging.info(
        "Valid Epoch {}, Loss_aligned {losses_cross:.3f} Loss_local {losses_local:.3f}".format(
            epoch, losses_cross=loss_cross, losses_local=loss_local))

    writer.add_scalar('val/loss_aligned', loss_cross, epoch)
    writer.add_scalar('val/loss_local', loss_local, epoch)

    logging.info("###### Valid Epoch {} End #####".format(epoch))

    return loss_cross, loss_local


def step_cross_model(data_loader, client_list, epoch, args, step_mode='train', debug=False):
    k = len(client_list)
    losses = utils.AverageMeter()
    loss_per_client = [[] for i in range(k)]
    # FISL
    logitis_sim_list = []
    logits_sim_target_list = []

    for step, (data_X, data_Y) in enumerate(data_loader):
        data_X = [data_X[idx] for idx in args.client_idx]
        if isinstance(data_X[0], dict) or isinstance(data_X[0], list):
            pass
        else:
            data_X = [x.float().to(args.device) for x in data_X]
        target = data_Y.view(-1).long().to(args.device)

        N = target.size(0)
        # features computed locally by each party
        exchanged_features = [client_list[i].get_exchanged_feature(data_X[i]) for i in range(k)]
        exchanged_features_for_transfer = [feature.detach().clone() for feature in exchanged_features]

        logitis_sim_list = [client_list[i].get_sim_feature(data_X[i]).to(args.device) for i in range(k)]
        logits_sim_target_list = [feature.detach().clone().to(args.device) for feature in logitis_sim_list]


        if args.pt_feat_iso_sigma > 0:
            with torch.no_grad():
                exchanged_features_for_transfer[0] = utils.encrypt_with_iso(exchanged_features_for_transfer[0],
                                                                              args.pt_feat_iso_sigma,
                                                                              args.pt_iso_threshold,
                                                                              args.device)

        loss_total = 0
        for i, client in enumerate(client_list):
            logits_sim_target_avg_list = []
            for f in range(k):
                logits_sim_target_avg_list.append(logits_sim_target_list[f])
            logits_sim_target_avg = torch.mean(torch.stack(logits_sim_target_avg_list), 0).to(args.device)
            if i == 0:
                # for the active party (the party has label).
                # The active party receives features from all other parties.
                logits_sim = logitis_sim_list[i].to(args.device)
                # inputs = F.softmax(logits_sim, dim=1)
                inputs = logits_sim
                # inputs = F.normalize(logits_sim)
                # targets = F.softmax(logits_sim_target_avg, dim=1)
                targets = logits_sim_target_avg
                # targets = F.normalize(logits_sim_target_avg)
                if step_mode == 'train':
                    loss, cross_meta = client.train_cross_model(data_X[i], target, exchanged_features[i],
                                                                exchanged_features_for_transfer[1:], epoch, inputs, targets)
                else:
                    loss, cross_meta = client.valid_cross_model(data_X[i], target, exchanged_features[i],
                                                                exchanged_features_for_transfer[1:], epoch, inputs, targets)
            else:
                logits_sim = logitis_sim_list[i].to(args.device)
                inputs = logits_sim
                # inputs = F.normalize(logits_sim)
                targets = logits_sim_target_avg
                # targets = F.normalize(logits_sim_target_avg)


                # for the passive party (the party has no label).
                # The passive party receives features only from the active party
                if step_mode == 'train':
                    loss, cross_meta = client.train_cross_model(data_X[i], None, exchanged_features[i],
                                                                exchanged_features_for_transfer[0], epoch, inputs, targets)
                else:
                    loss, cross_meta = client.valid_cross_model(data_X[i], None, exchanged_features[i],
                                                                exchanged_features_for_transfer[0], epoch, inputs, targets)
            loss_total = loss_total + loss
            loss_per_client[i].append(loss)

        losses.update(loss_total / k, N)
    loss_per_client = [sum(item) / len(item) for item in loss_per_client]
    meta = {'loss_per_client': loss_per_client}
    return losses.avg, meta


def step_local_model(data_loader, client_list, epoch, args, step_mode='train'):
    k = len(client_list)

    losses = utils.AverageMeter()
    loss_per_client = None
    loss_per_client_reg = None
    for local_epoch in range(args.local_epochs_local):
        # only record the last local epoch
        loss_per_client = [[] for i in range(k)]
        loss_per_client_reg = [[] for i in range(k)]

        for step, (data_X, data_Y) in enumerate(data_loader):
            data_X = [data_X[idx] for idx in args.client_idx]

            if isinstance(data_X[0], dict) or isinstance(data_X[0], list):
                pass
            else:
                data_X = [x.float().to(args.device) for x in data_X]
            target = data_Y.view(-1).long().to(args.device)

            N = target.size(0)
            loss_total = 0
            # local SSL
            for i, client in enumerate(client_list):
                if step_mode == 'train':
                    loss, local_meta = client.train_local_model(data_X[i], target, epoch)
                else:
                    loss, local_meta = client.valid_local_model(data_X[i], target, epoch)
                loss_per_client_reg[i].append(local_meta['loss_debug'])

                loss_per_client[i].append(loss)
                loss_total = loss_total + loss
            if local_epoch == args.local_epochs_local - 1:
                losses.update(loss_total / k, N)
        loss_per_client = [sum(item) / len(item) for item in loss_per_client]
        loss_per_client_reg = [sum(item) / len(item) for item in loss_per_client_reg]

    meta = {'loss_per_client': loss_per_client, 'loss_per_client_reg': loss_per_client_reg}
    return losses.avg, meta


if __name__ == '__main__':
    main()
