import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .resnet import resnet18
import copy


class MyResnet18(nn.Module):

    def __init__(self, class_num=10, output_dim=512):
        super(MyResnet18, self).__init__()
        model = resnet18(pretrained=False, num_classes=class_num, output_dim=output_dim)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [64, 64, 56, 56]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BottomResnet18(nn.Module):
    def __init__(self):
        super(BottomResnet18, self).__init__()
        model = resnet18(pretrained=False)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [64, 64, 56, 56]
        x = self.layer1(x)  # [64, 64, 56, 56]
        return x


class TopResnet18(nn.Module):

    def __init__(self, class_num=10, output_dim=512):
        super(TopResnet18, self).__init__()
        model = resnet18(pretrained=False, num_classes=class_num, output_dim=output_dim)
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 512]):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return x


class BottomMLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(BottomMLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


class TopMLP2(nn.Module):
    def __init__(self, hidden_dims=[512, 512]):
        super(TopMLP2, self).__init__()
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer2(x)
        return x


class ClassificationModelHost(nn.Module):

    def __init__(self, encoder_local_bottom, encoder_local_top, encoder_cross, hidden_dim, num_classes,
                 use_encoder_cross=False, use_encoder_local=False, pool='mean', ratio=0.5, mlp_layer=1):
        super().__init__()
        self.ratio = ratio
        self.pool = pool
        self.use_encoder_cross = use_encoder_cross
        self.use_encoder_local = use_encoder_local
        self.backbone = nn.ModuleList()
        hidden_dim_ratio = 1
        if self.pool == 'concat':
            if use_encoder_local is True and use_encoder_cross is True:
                hidden_dim_ratio = 2
        hidden_dim = hidden_dim * hidden_dim_ratio
        if self.use_encoder_cross:
            self.encoder_cross = copy.deepcopy(encoder_cross)
            self.backbone.append(self.encoder_cross)

        if self.use_encoder_local:
            self.encoder_local_bottom = copy.deepcopy(encoder_local_bottom)
            self.encoder_local_top = copy.deepcopy(encoder_local_top)
            self.backbone.append(self.encoder_local_bottom)
            self.backbone.append(self.encoder_local_top)

        if mlp_layer == 1:
            self.classifier_head = nn.Linear(hidden_dim, num_classes)
        elif mlp_layer == 2:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                nn.ReLU(),
                nn.Linear(hidden_dim[1], num_classes)
            )
        elif mlp_layer == 3:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                nn.ReLU(),
                nn.Linear(hidden_dim[1], hidden_dim[2]),
                nn.ReLU(),
                nn.Linear(hidden_dim[2], num_classes)
            )

            # 添加门控参数
        self.update_gate = nn.Sequential(
                nn.Linear(512 * 2, 512),  # 假设x_cross和x_local的维度都是hidden_dim
                nn.Sigmoid()
            )
        self.candidate_layer = nn.Sequential(
                nn.Linear(512 * 2, 512),
                # nn.Tanh()
            )
        self.reset_gate = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.Sigmoid()
        )


    def forward(self, input_X):
        if self.use_encoder_cross:
            x_cross = self.encoder_cross(input_X).flatten(start_dim=1)
        if self.use_encoder_local:
            f = self.encoder_local_bottom(input_X)
            x_local = self.encoder_local_top(f).flatten(start_dim=1)
        if self.use_encoder_cross and self.use_encoder_local:
            if self.pool == 'mean':

                combined = torch.cat((x_cross, x_local), dim=1)
                update = self.update_gate(combined)
                reset = self.reset_gate(combined)
                # Apply reset gate to previous state (x_cross)
                x_cross_reset = x_cross * reset
                # x_local_reset = x_local * reset
                # Prepare combined reset state with current input/state for candidate calculation
                # combined_reset = torch.cat((x_local_reset, x_cross), dim=1)
                combined_reset = torch.cat((x_cross_reset, x_local), dim=1)
                candidate = self.candidate_layer(combined_reset)
                # Final state is a combination of the old state and the new candidate state, modulated by the update gate
                z = update * candidate + (1 - update) * x_local
            # non-reset-gate-method
            # combined = torch.cat([x_cross, x_local], dim=1)
            # update = self.update_gate(combined)
            # candidate = self.candidate_layer(combined)
            # z = update * candidate + (1 - update) * x_local


            elif self.pool == 'concat':
                z = torch.cat([x_cross, x_local], dim=1)
            else:
                z = x_cross * self.ratio + x_local * (1 - self.ratio)
        elif self.use_encoder_cross:
                z = x_cross
        elif self.use_encoder_local:
                z = x_local
        else:
            raise Exception
        return z

    def get_prediction(self, z_0, z_list):
        if z_list is not None:
            out = torch.cat([z_0] + z_list, dim=1)
        else:
            out = z_0
        x = self.classifier_head(out)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_encoder_cross(self, load_path, device):
        self.encoder_cross.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_top(self, load_path, device):
        self.encoder_local_top.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_bottom(self, load_path, device):
        self.encoder_local_bottom.load_state_dict(torch.load(load_path, map_location=device))


class ClassificationModelGuest(nn.Module):

    def __init__(self, encoder_local_bottom, encoder_local_top, encoder_cross, use_encoder_cross=False, use_encoder_local=False,
                 pool='mean', ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.pool = pool
        self.use_encoder_cross = use_encoder_cross
        self.use_encoder_local = use_encoder_local
        self.backbone = nn.ModuleList()

        if self.use_encoder_cross:
            self.encoder_cross = copy.deepcopy(encoder_cross)
            self.backbone.append(self.encoder_cross)

        if self.use_encoder_local:
            self.encoder_local_bottom = copy.deepcopy(encoder_local_bottom)
            self.encoder_local_top = copy.deepcopy(encoder_local_top)
            self.backbone.append(self.encoder_local_bottom)
            self.backbone.append(self.encoder_local_top)

        self.update_gate = nn.Sequential(
            nn.Linear(512 * 2, 512),  # 假设x_cross和x_local的维度都是hidden_dim
            nn.Sigmoid()
        )
        self.candidate_layer = nn.Sequential(
            nn.Linear(512 * 2, 512),
            # nn.Tanh()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.Sigmoid()
        )


    def forward(self, input_X):
        if self.use_encoder_cross:
            x_cross = self.encoder_cross(input_X).flatten(start_dim=1)
        if self.use_encoder_local:
            f = self.encoder_local_bottom(input_X)
            x_local = self.encoder_local_top(f).flatten(start_dim=1)
        if self.use_encoder_cross and self.use_encoder_local:
            if self.pool == 'mean':
                combined = torch.cat((x_cross, x_local), dim=1)
                update = self.update_gate(combined)
                reset = self.reset_gate(combined)

                # Apply reset gate to previous state (x_cross)
                x_cross_reset = x_cross * reset
                # x_local_reset = x_local * reset
                # Prepare combined reset state with current input/state for candidate calculation
                # combined_reset = torch.cat((x_local_reset, x_cross), dim=1)
                combined_reset = torch.cat((x_cross_reset, x_local), dim=1)
                candidate = self.candidate_layer(combined_reset)
                # Final state is a combination of the old state and the new candidate state, modulated by the update gate
                z = update * candidate + (1 - update) * x_local

            # non-reset-gate-method
            # combined = torch.cat([x_cross, x_local], dim=1)
            # update = self.update_gate(combined)
            # candidate = self.candidate_layer(combined)
            # z = update * candidate + (1 - update) * x_local

            elif self.pool == 'concat':
                z = torch.cat([x_cross, x_local], dim=1)
            else:
                z = x_cross * self.ratio + x_local * (1 - self.ratio)
        elif self.use_encoder_cross:
            z = x_cross
        elif self.use_encoder_local:
            z = x_local
        else:
            raise Exception
        return z

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_encoder_cross(self, load_path, device):
        self.encoder_cross.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_top(self, load_path, device):
        self.encoder_local_top.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_bottom(self, load_path, device):
        self.encoder_local_bottom.load_state_dict(torch.load(load_path, map_location=device))

    def save_models(self, target_dir, name_str):
        os.makedirs(target_dir, exist_ok=True)

        if self.use_encoder_cross:
            torch.save(self.encoder_cross.state_dict(),
                       os.path.join(target_dir, 'model_encoder_cross-{}.pth'.format(name_str)))
        if self.use_encoder_local:
            torch.save(self.encoder_local_bottom.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_bottom-{}.pth'.format(name_str)))
            torch.save(self.encoder_local_top.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_top-{}.pth'.format(name_str)))