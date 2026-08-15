"""
part of the code adopted from https://github.com/lightly-ai/lightly
"""
import torch.nn as nn
import torch.utils
import torch
import copy

from models.ssl_utils import NTXentLoss, projection_mlp, NegCosineSimilarityLoss
from pretrain_template import ClientTemplate


def get_clients(encoder_local_top_bottom_list, encoder_local_top_list, encoder_cross_list, args):
    client_list = []
    for i in range(args.k):
        client = SSClient(i, [encoder_local_top_bottom_list[i], encoder_local_top_list[i], encoder_cross_list[i]], args)
        client_list.append(client)
    return client_list


class SSClient(ClientTemplate):

    def __init__(self, client_idx, models, args):
        super(SSClient, self).__init__(client_idx, models, args)

        # methods related models and optimizers
        self.projection_mlp_cross = projection_mlp(args.num_ftrs, self.proj_hidden_dim, self.out_dim,
                                                         self.num_mlp_layers).to(args.device)
        self.projection_mlp_local = projection_mlp(args.num_ftrs, self.proj_hidden_dim, self.out_dim,
                                                         self.num_mlp_layers).to(args.device)

        # loss
        self.cross_criterion_sim = NegCosineSimilarityLoss().to(args.device)
        self.cross_criterion = NTXentLoss().to(args.device)
        self.local_criterion_one = NTXentLoss(memory_bank_size=4096).to(args.device)
        self.local_criterion_two = NTXentLoss(memory_bank_size=4096).to(args.device)
        self.local_criterion_constraint = NTXentLoss().to(args.device)

        # setting for aggregation
        self.model_local_top = nn.ModuleList(
            [self.encoder_local_top, self.projection_mlp_local])

        self.models = nn.ModuleList([self.encoder_cross, self.projection_mlp_cross,
                                     self.encoder_local_bottom, self.encoder_local_top, self.projection_mlp_local])

        # -byol-simloss-seed0
        self.encoder_local_bottom_momentum = copy.deepcopy(self.encoder_local_bottom)
        self.encoder_local_top_momentum = copy.deepcopy(self.encoder_local_top)
        self.projection_head_local_momentum = copy.deepcopy(self.projection_mlp_local)

        deactivate_requires_grad(self.encoder_local_bottom_momentum)
        deactivate_requires_grad(self.encoder_local_top_momentum)
        deactivate_requires_grad(self.projection_head_local_momentum)

        # rescale learning rate
        self.pretrain_lr_ratio = 0.5/(self.args.local_ratio + self.args.constraint_ratio)

        # get optimizer
        self.optimizer_list_cross = [self.get_optimizer(self.encoder_cross, self.args.optimizer),
                                     self.get_optimizer(self.projection_mlp_cross)]

        self.optimizer_list_local = [self.get_optimizer(self.encoder_local_bottom, self.args.optimizer),
                                     self.get_optimizer(self.encoder_local_top, self.args.optimizer),
                                     self.get_optimizer(self.projection_mlp_local)]

        # learning rate scheduler
        self.scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            for optimizer in self.optimizer_list_cross[:-1] if optimizer is not None] +[
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            for optimizer in self.optimizer_list_local[:-1] if optimizer is not None]

        self.model_to_device(args.device)

    def get_local_projection_momentum_feature(self, x):
        if isinstance(x, list):
            f_local = self.encoder_local_bottom_momentum(x[0].float().to(self.args.device))
            h_local = self.encoder_local_top_momentum(f_local).flatten(start_dim=1)
            z_local = self.projection_head_local_momentum(h_local)
        else:
            f_local = self.encoder_local_bottom_momentum(x)
            h_local = self.encoder_local_top_momentum(f_local).flatten(start_dim=1)
            z_local = self.projection_head_local_momentum(h_local)
        z_local = z_local.detach()
        return z_local

    def compute_cross_loss(self, x, y, z_cross_own, z_cross_received, epoch, sim=None, logits_sim_target_avg=None):
        meta = {}
        loss_debug = []

        if isinstance(z_cross_received, list):
            loss = torch.tensor(0).to(self.args.device)
            for z_item in z_cross_received:
                loss_ind = self.cross_criterion(z_cross_own, z_item.detach())
                loss_debug.append(loss_ind.item())
                loss = loss + loss_ind
            loss = loss / len(z_cross_received)
            if self.args.use_FISL != 0:
                loss_sim = self.cross_criterion_sim(sim, logits_sim_target_avg.detach())
                # loss_sim = self.cross_criterion(sim, logits_sim_target_avg.detach())
                loss = loss + self.args.dis_power*loss_sim
        else:
            loss = self.cross_criterion(z_cross_own, z_cross_received.detach())
            if self.args.use_FISL != 0:
                # loss_sim = self.cross_criterion(sim, logits_sim_target_avg.detach())
                loss_sim = self.cross_criterion_sim(sim, logits_sim_target_avg.detach())
                loss = loss + self.args.dis_power*loss_sim
        meta['loss_debug'] = loss_debug
        return loss, loss_debug

    def compute_local_loss(self, x, y, epoch=0):
        meta = {}
        loss_constraint = torch.tensor(0).to(self.args.device)

        update_momentum(self.encoder_local_bottom, self.encoder_local_bottom_momentum, m=0.99)
        update_momentum(self.encoder_local_top, self.encoder_local_top_momentum, m=0.99)
        update_momentum(self.projection_mlp_local, self.projection_head_local_momentum, m=0.99)

        # two augemented views
        x1 = x[0].float().to(self.args.device)
        x2 = x[1].float().to(self.args.device)

        # model outputs of two augemented views
        f1 = self.encoder_local_bottom(x1)
        h1 = self.encoder_local_top(f1).flatten(start_dim=1)
        z1 = self.projection_mlp_local(h1)

        f2 = self.encoder_local_bottom(x2)
        h2 = self.encoder_local_top(f2).flatten(start_dim=1)
        z2 = self.projection_mlp_local(h2)
        z1_momentum = self.get_local_projection_momentum_feature(x1)
        z2_momentum = self.get_local_projection_momentum_feature(x2)

        # local contrastive loss using augmented views
        loss = self.args.local_ratio * (self.local_criterion_one(z1, z2_momentum.detach()) +
                              self.local_criterion_two(z2, z1_momentum.detach()))

        # regularization loss of cross encoder
        if self.args.constraint_ratio > 0:
            h1_cross = self.encoder_cross(x1).flatten(start_dim=1)
            h2_cross = self.encoder_cross(x2).flatten(start_dim=1)
            z1_cross = self.projection_mlp_cross(h1_cross)
            z2_cross = self.projection_mlp_cross(h2_cross)

            loss_constraint = self.args.constraint_ratio * (self.local_criterion_one(z1, z1_cross.detach()) +
                                                            self.local_criterion_two(z2, z2_cross.detach()))
            loss = loss + loss_constraint



        meta['loss_debug'] = loss_constraint.item()
        return loss, meta


def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False


def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`"""
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1. - m)