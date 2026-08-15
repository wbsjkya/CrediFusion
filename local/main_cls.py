import os
import sys
import time
import torch
import copy
import glob
import numpy as np
from datetime import datetime
import logging
import torch.utils
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from prepare_experiments import get_dataset, get_model, prepare_exp, set_random_seed, save_models_from_passive_client
from models.model_templates import ClassificationModelGuest, ClassificationModelHost
import utils


def main():
    # read args
    args, tb_writer = prepare_exp('cls')

    # set seed
    set_random_seed(args.seed)

    # set gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logging.info('gpu device = %d' % args.gpu)

    # get dataloader
    train_loader_aligned, train_loader_local, valid_loader, test_loader, args = get_dataset(args)

    # get model modules
    encoder_models_local_bottom, encoder_models_local_top, encoder_models_cross, args = get_model(args)

    # save experiment variables
    if 'ctr' in args.dataset:
        args_log = copy.deepcopy(args)
        args_log.col_names = ''
        logging.info("args = {}".format(vars(args_log)))
    else:
        logging.info("args = {}".format(vars(args)))

    use_local_model = False
    use_cross_model = False

    # if use pretrained model and use local data and flag of use_local_model is set
    if args.use_local_model and args.pretrained_path != '' and args.local_ssl != 0:
        use_local_model = True
    #
    if args.use_cross_model:
        use_cross_model = True


    logging.info("models used: cross {}, local {}".format(use_cross_model, use_local_model))

    model_list = []
    # prepare training models and optimizers
    clsmodel_main = ClassificationModelHost(copy.deepcopy(encoder_models_local_bottom[0]),
                                                 copy.deepcopy(encoder_models_local_top[0]),
                                                 copy.deepcopy(encoder_models_cross[0]),
                                                 args.num_ftrs * args.k, args.num_classes,
                                                 use_cross_model, use_local_model, args.pool, 0.5, args.num_cls_layer)

    model_list.append(clsmodel_main)
    for i in range(args.k - 1):
        guest_model = ClassificationModelGuest(copy.deepcopy(encoder_models_local_bottom[i + 1]),
                                               copy.deepcopy(encoder_models_local_top[i + 1]),
                                               copy.deepcopy(encoder_models_cross[i + 1]),
                                               use_cross_model, use_local_model, args.pool)
        model_list.append(guest_model)

    encoder_models_local_bottoms = None
    encoder_models_local_top = None
    encoder_models_cross = None

    model_list = [model.to(args.device) for model in model_list]

    if args.pretrained_path != '':
        for i in range(args.k):
            if use_cross_model:
                model_list[i].load_encoder_cross(
                    './{}/model_encoder_cross-'.format(args.pretrain_model_dir) + args.pretrained_path + '-{}.pth'.format(i),
                    args.device)
            if use_local_model:
                model_list[i].load_encoder_local_bottom('./{}/model_encoder_local_bottom-'.format(args.pretrain_model_dir) +
                                             args.pretrained_path + '-{}.pth'.format(i), args.device)
                model_list[i].load_encoder_local_top(
                    './{}/model_encoder_local_top-'.format(args.pretrain_model_dir)+ args.pretrained_path + '-{}.pth'.format(i),
                    args.device)

        logging.info('***** USE PRETRAIN MODEL： {}, {}'.format(args.pretrained_path, args.pretrained_path))

    if args.freeze_backbone == 1:
        # all model backbone freezed
        for model in model_list:
            model.freeze_backbone()
        logging.info('***** FREEZE BACKBONE')
    elif args.freeze_backbone == 2:
        # first model is active
        for model in model_list[1:]:
            model.freeze_backbone()
        logging.info('***** FREEZE BACKBONE, EXCEPT THE FIRST')

    # criterion
    if "ctr" in args.dataset or 'bhi' in args.dataset:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    # weights optimizer
    if "ctr" in args.dataset:
        optimizer_list = [
            torch.optim.Adagrad(model.parameters(), args.learning_rate)
            for model in model_list]
    else:
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            for model in model_list]

    scheduler_list = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for optimizer in optimizer_list]
    best_acc_top1 = 0.

    for epoch in range(args.epochs):

        # training
        train_acc, train_obj = train(train_loader_aligned, model_list, optimizer_list, criterion, epoch, args, tb_writer)

        # validation
        cur_step = (epoch+1) * len(train_loader_aligned)
        valid_acc_top1, valid_obj = validate(test_loader, model_list, criterion, epoch, cur_step, args, tb_writer)

        # save
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
        logging.info('best_acc_top1 %f', best_acc_top1)
        for scheduler in scheduler_list:
            scheduler.step()

    # save models
    # save_models_from_passive_client(model_list[-1], args)
    logging.info("***** model saved *****")


def train(train_loader, model_list, optimizer_list, criterion, epoch, args, writer):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer_list[0].param_groups[0]['lr']
    logging.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    for model in model_list:
        model.train()

    k = len(model_list)

    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X = [trn_X[idx] for idx in args.client_idx]
        # dict for bert model
        if isinstance(trn_X[0], dict):
            pass
        else:
            trn_X = [x.float().to(args.device) for x in trn_X]

        target = trn_y.view(-1).long().to(args.device)

        N = target.size(0)
        z_rest_clone = None

        z_list = [model_list[i](trn_X[i]) for i in range(0, len(model_list))]
        z_0 = z_list[0]
        if k > 1:
            z_rest = z_list[1:]
            z_rest_clone = [z.detach().clone() for z in z_rest]
            z_rest_clone = [torch.autograd.Variable(z, requires_grad=True).to(args.device) for z in z_rest_clone]

        logits = model_list[0].get_prediction(z_0, z_rest_clone)
        if 'ctr' in args.dataset or 'bhi' in args.dataset:
            loss = criterion(torch.sigmoid(logits.view(-1)), target.float())
        else:
            loss = criterion(logits, target)
        if k > 1:
            if args.freeze_backbone == 0:
                z_gradients_list = [torch.autograd.grad(loss, z, retain_graph=True) for z in z_rest_clone]
                if args.cls_iso_sigma > 0:
                    z_gradients_list = [utils.encrypt_with_iso(z[0], args.cls_iso_sigma, args.cls_iso_threshold, args.device)
                                              for z in z_gradients_list]

                weights_gradients_list = [
                    torch.autograd.grad(z_rest[i], model_list[i + 1].parameters(), grad_outputs=z_gradients_list[i],
                                        retain_graph=True) for i in range(len(z_gradients_list))]

        optimizer_list[0].zero_grad()
        loss.backward()  # retain_graph=True)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
        optimizer_list[0].step()

        if k > 1:
            if args.freeze_backbone == 0:
                [optimizer_list[i].zero_grad() for i in range(1, k)]
                for i in range(len(weights_gradients_list)):
                    for w, g in zip(model_list[i + 1].parameters(), weights_gradients_list[i]):
                        if w.requires_grad:
                            w.grad = g.detach()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)
                    optimizer_list[i + 1].step()
        if "mosei" in args.dataset:
            prec1 = [torch.tensor([0])]
        else:
            prec1 = utils.accuracy(logits, target, topk=(1,))

        losses.update(loss.item(), N)
        top1.update(prec1[0].item(), N)

        if step % args.report_freq == 0 or step == len(train_loader) - 1:
            logging.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1f}%)".format(
                    epoch + 1, args.epochs, step, len(train_loader) - 1, losses=losses, top1=top1))

        writer.add_scalar('train/loss', losses.avg, cur_step)
        writer.add_scalar('train/top1', top1.avg, cur_step)
        cur_step += 1

    return top1.avg, losses.avg


def validate(valid_loader, model_list, criterion, epoch, cur_step, args, writer):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    for model in model_list:
        model.eval()

    k = len(model_list)

    y_gt_list = []
    y_pred_list = []
    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_loader):
            val_X = [val_X[idx] for idx in args.client_idx]
            if isinstance(val_X[0], dict):
                pass
            else:
                val_X = [x.float().to(args.device) for x in val_X]
            target = val_y.view(-1).long().to(args.device)
            N = target.size(0)

            z_rest_clone = None

            z_list = [model_list[i](val_X[i]) for i in range(0, len(model_list))]
            z_0 = z_list[0]
            if k > 1:
                z_rest = z_list[1:]
                z_rest_clone = [z.detach().clone() for z in z_rest]
                z_rest_clone = [torch.autograd.Variable(z, requires_grad=True).to(args.device) for z in
                                z_rest_clone]

            logits = model_list[0].get_prediction(z_0, z_rest_clone)
            if 'ctr' in args.dataset or 'bhi' in args.dataset:
                loss = criterion(torch.sigmoid(logits.view(-1)), target.float())
            else:
                loss = criterion(logits, target)

            if 'ctr' in args.dataset or 'bhi' in args.dataset:
                y_gt_list.append(target.float().cpu().numpy())
                y_pred_list.append(torch.sigmoid(logits.view(-1).cpu()).numpy())
                prec1 = [torch.tensor(0)]
            else:
                prec1 = utils.accuracy(logits, target, topk=(1,))

            losses.update(loss.item(), N)
            top1.update(prec1[0].item(), N)

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)

    # auc for ctr dataset
    if 'ctr' in args.dataset:
        y_gt = np.concatenate(y_gt_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        score = roc_auc_score(y_gt, y_pred)
        logging.info(
            "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
            "Prec@(1,5) ({:.3f})".format(
                epoch + 1, args.epochs, step, len(valid_loader) - 1, losses.avg, score))
        return score, losses.avg
    # f1 score for bhi dataset
    elif 'bhi' in args.dataset:
        y_gt = np.concatenate(y_gt_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        precision, recall, thresholds = precision_recall_curve(y_gt, y_pred)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        score = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]

        logging.info(
            "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
            "Prec@(1,5) ({:.3f})".format(
                epoch + 1, args.epochs, step, len(valid_loader) - 1, losses.avg, score))
        return score, losses.avg
    # acc for other datasets
    else:
        logging.info(
            "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
            "Prec@(1,5) ({top1.avg:.1f}%)".format(
                epoch + 1, args.epochs, step, len(valid_loader) - 1, losses=losses, top1=top1))
        return top1.avg, losses.avg


if __name__ == '__main__':
    main()


