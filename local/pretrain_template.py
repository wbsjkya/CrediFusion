"""
part of the code adopted from https://github.com/lightly-ai/lightly
"""

import os
import torch.nn as nn
import torch.utils
import torch
import copy

import utils
import torch.nn.functional as F

def get_clients(encoder_local_bottom_list, encoder_local_list, encoder_cross_list, args):
    client_list = []
    for i in range(args.k):
        client = ClientTemplate(i, [encoder_local_bottom_list[i], encoder_local_list[i], encoder_cross_list[i]], args)
        client_list.append(client)
    return client_list

# def off_diagonal(x):
#     n, m = x.shape
#     assert n == m
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
class ClientTemplate():

    def __init__(self, client_idx, models, args):
        self.client_idx = client_idx
        self.args = args
        self.device = args.device

        # settings for projector and predictor
        self.out_dim = args.out_dim
        self.proj_hidden_dim = args.proj_hidden_dim
        self.pred_hidden_dim = args.pred_hidden_dim
        self.num_mlp_layers = args.proj_layer

        # main models and optimizers
        self.encoder_local_bottom = copy.deepcopy(models[0]).to(args.device)
        self.encoder_local_top = copy.deepcopy(models[1]).to(args.device)
        self.encoder_cross = copy.deepcopy(models[2]).to(args.device)

        self.models = nn.ModuleList()
        self.model_local_top = nn.ModuleList()
        self.model_local = nn.ModuleList()


        # rescale learning rate
        self.pretrain_lr_ratio = 0.5/(self.args.local_ratio + self.args.constraint_ratio)

        # optimizer list
        self.optimizer_list_cross = []
        self.optimizer_list_local = []

        # learning rate scheduler
        self.scheduler_list = []
        # 11
        self.client_pre_model = copy.deepcopy(self.model_local)



    def model_to_device(self, device):
        for model in self.models:
            model.to(device)

    def set_train(self):
        for model in self.models:
            model.train()

    def set_eval(self):
        for model in self.models:
            model.eval()

    def get_exchanged_feature(self, x):
        if isinstance(x, list):
            h_cross = self.encoder_cross(x[0].float().to(self.args.device)).flatten(start_dim=1)
            z_cross = self.projection_mlp_cross(h_cross)
        else:
            h_cross = self.encoder_cross(x).flatten(start_dim=1)
            z_cross = self.projection_mlp_cross(h_cross)
        return z_cross

    def _calculate_isd_sim(self, features):
        sim_q = torch.mm(features, features.T)
        logits_mask = torch.scatter(
            torch.ones_like(sim_q),
            1,
            torch.arange(sim_q.size(0)).view(-1, 1).to(self.device),
            0
        )
        row_size = sim_q.size(0)
        sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
        return sim_q / self.args.temp

    def get_sim_feature(self, x):
        if isinstance(x, list):
            h_cross = self.encoder_cross(x[0].float().to(self.args.device)).flatten(start_dim=1)
            # z_cross = self.projection_mlp_cross(h_cross)
            features = F.normalize(h_cross, dim=1)
            # features = F.normalize(z_cross, dim=1)
        else:
            h_cross = self.encoder_cross(x).flatten(start_dim=1)
            # z_cross = self.projection_mlp_cross(h_cross)
            features = F.normalize(h_cross, dim=1)
            # features = F.normalize(z_cross, dim=1)
        logits_sim = self._calculate_isd_sim(features)

        return logits_sim

    def adjust_learning_rate(self):
        for scheduler in self.scheduler_list:
            scheduler.step()

    def get_optimizer(self, model, opt_type='sgd'):
        pretrain_lr_head = self.args.pretrain_lr_head * self.args.batch_size / 256
        pretrain_lr_encoder = self.args.pretrain_lr_encoder * self.args.batch_size / 256
        if opt_type == 'sgd':
            return torch.optim.SGD(model.parameters(), pretrain_lr_head * self.pretrain_lr_ratio,
                                   momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif opt_type == 'adagrad':
            return torch.optim.Adagrad(model.parameters(), pretrain_lr_encoder * self.pretrain_lr_ratio)
        else:
            return None

    def opt_preprocess(self, submodel='cross'):
        if submodel == 'cross':
            for opt in self.optimizer_list_cross:
                opt.zero_grad()
        elif submodel == 'local':
            for opt in self.optimizer_list_local:
                opt.zero_grad()

    def opt_postprocess(self, submodel='cross'):
        if submodel == 'cross':
            for opt in self.optimizer_list_cross:
                opt.step()
        elif submodel == 'local':
            for opt in self.optimizer_list_local:
                opt.step()

    def compute_cross_loss(self, x, y, z_cross_own, z_cross_received, epoch, sim=None, logits_sim_target_avg=None):
        pass

    def compute_local_loss(self, x, y, epoch=0):
        pass

    def train_cross_model(self, x, y, z_cross_own, z_cross_received,  epoch, sim=None, logits_sim_target_avg=None):
        loss_total = []
        for local_epoch in range(self.args.local_epochs_cross):
            if local_epoch > 0:
                z_cross_own = self.get_exchanged_feature(x)

            self.opt_preprocess('cross')
            loss, cross_meta = self.compute_cross_loss(x, y, z_cross_own, z_cross_received, epoch, sim, logits_sim_target_avg)

            # if self.args.use_FISL != 0:
            #     loss_distill = F.kl_div(sim, logits_sim_target_avg, reduction='batchmean')
            #     loss = loss + self.args.dis_power*loss_distill
            # torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            self.opt_postprocess('cross')

            loss_total.append(loss.item())

        loss_mean = sum(loss_total) / len(loss_total)
        return loss_mean, cross_meta

    def train_local_model(self, x, y, epoch):
        self.opt_preprocess('local')
        loss, local_meta = self.compute_local_loss(x, y, epoch)
        loss.backward()
        # gradient clip
        if self.args.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.models.parameters(), self.args.grad_clip)

        # update model
        self.opt_postprocess('local')

        return loss.item(), local_meta

    def update_local_top_model(self, backbone_local_state_dict, x=None, device='cpu'):
        if x is None:
            self.model_local_top.load_state_dict(backbone_local_state_dict)
            self.model_local_top.to(device)
        else:
            self.model_local_top.load_state_dict(backbone_local_state_dict)
            self.model_local_top.to(device)

    def update_local_model(self, backbone_local_state_dict, x=None, device='cpu'):
        if x is None:
            self.client_pre_model.load_state_dict(backbone_local_state_dict)
            self.client_pre_model.to(device)
        else:
            self.client_pre_model.load_state_dict(backbone_local_state_dict)
            self.client_pre_model.to(device)


    def get_local_top_model(self, defense_ratio=0.0):
        if defense_ratio > 0.0:
            ret_model = copy.deepcopy(self.model_local_top)
               
            with torch.no_grad():
                for param in ret_model.parameters():
                    param.data = utils.encrypt_with_iso(param.data, defense_ratio)
            return ret_model.cpu().state_dict()

        return self.model_local_top.cpu().state_dict()

    def get_local_model(self, defense_ratio=0.0):
        if defense_ratio > 0.0:
            ret_model = copy.deepcopy(self.model_local)

            with torch.no_grad():
                for param in ret_model.parameters():
                    param.data = utils.encrypt_with_iso(param.data, defense_ratio)
            return ret_model.cpu().state_dict()

        return self.model_local.cpu().state_dict()

    def valid_cross_model(self, x, y, z_cross_own, z_cross_received, epoch, sim=None, logits_sim_target_avg=None):
        loss, cross_meta = self.compute_cross_loss(x, y, z_cross_own, z_cross_received, epoch, sim, logits_sim_target_avg)
        # if self.args.use_FISL != 0:
        #     loss_distill = F.kl_div(sim, logits_sim_target_avg, reduction='batchmean')
        #     loss = loss + self.args.dis_power * loss_distill
        return loss.item(), cross_meta

    def valid_local_model(self, x, y, epoch):
        loss, local_meta = self.compute_local_loss(x, y, epoch)
        return loss.item(), local_meta

    def save_models(self, target_dir, name_str, idx):
        os.makedirs(target_dir, exist_ok=True)
        torch.save(self.encoder_cross.state_dict(),
                   os.path.join(target_dir, 'model_encoder_cross-{}-{}.pth'.format(name_str, idx)))
        if self.args.local_ssl:
            torch.save(self.encoder_local_bottom.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_bottom-{}-{}.pth'.format(name_str, idx)))

            torch.save(self.encoder_local_top.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_top-{}-{}.pth'.format(name_str, idx)))

    def get_cross_projection_feature(self, h):
        z_cross = self.projection_mlp_cross(h)
        return z_cross

    def get_cross_prediction_feature(self, z):
        p_cross = self.prediction_mlp_cross(z)
        return p_cross

    def get_local_projection_feature(self, h):
        z_local = self.projection_mlp_local(h)
        return z_local

    def get_local_prediction_feature(self, z):
        p_local = self.prediction_mlp_local(z)
        return p_local

    def get_cross_encoder_feature(self, x):
        if isinstance(x, list):
            h_cross = self.encoder_cross(x[0].float().to(self.args.device)).flatten(start_dim=1)
        else:
            h_cross = self.encoder_cross(x).flatten(start_dim=1)
        return h_cross

    def get_local_encoder_feature(self, x):
        if isinstance(x, list):
            f_local = self.encoder_local_bottom(x[0].float().to(self.args.device))
            h_local = self.encoder_local_top(f_local).flatten(start_dim=1)
        else:
            f_local = self.encoder_local_bottom(x)
            h_local = self.encoder_local_top(f_local).flatten(start_dim=1)
        return h_local


def aggregate_fedavg(w_locals):
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params


def get_server(models, dataloader, args):
    s = SSServer(models, dataloader, args)
    return s


class SSServer(object):

    def __init__(self, models, dataloader, args):
        self.args = args
        self.scheduler_list = []

    def aggregation(self, client_list, sample_num):
        k = len(client_list)
        loss_debug = []
        if self.args.aggregation_mode == 'pma':
            w_locals = []
            for idx, client in enumerate(client_list):
                # update dataset
                w = client.get_local_top_model()
                if isinstance(sample_num, list):
                    w_locals.append((sample_num[idx], copy.deepcopy(w)))
                else:
                    w_locals.append((sample_num, copy.deepcopy(w)))

            # update client_pre_model
            # if self.args.constraint_pre_ratio > 0:
            #     for i in range(len(client_list)):
            #         w = client_list[i].get_local_model()
            #         client_list[i].update_local_model(w, None, self.args.device) # update pre model
            #     print('client_pre_model update')


            # aggregate local models
            global_model_state_dict = aggregate_fedavg(w_locals)


            # update local model models
            for i in range(len(client_list)):
                client_list[i].update_local_top_model(global_model_state_dict, None, self.args.device)
        return loss_debug

    def adjust_learning_rate(self):
        for scheduler in self.scheduler_list:
            scheduler.step()
