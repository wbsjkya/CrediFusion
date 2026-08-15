import torch as t
from torch import nn
from torch.utils.data import Dataset
import logging
from typing import Optional
from typing import Any, Dict, List, Union, Callable, Literal
from dataclasses import dataclass, fields
logger = logging.getLogger(__name__)
from ssl_utils import NegCosineSimilarityLoss, NTXentLoss

'''
第一阶段，进行SSL：采用异质网络结构，将Top model作为平均汇聚处，Loss计算各方数据的差值，只训练bottom
第二阶段，本地进行训练，与第一阶段模型进行蒸馏
第三阶段，训练异质模型，bottom取Bl，先加载权重
'''
######################################################
# 自监督训练类覆写
######################################################

from transformers import PreTrainedTokenizer
from transformers import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from fate.arch import Context
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost, StdAggLayerArgument, Args, SSHEArgument, FedPassArgument, TopModelStrategyArguments
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerGuest, AggLayerHost
from fate.ml.nn.trainer.trainer_base import HeteroTrainerBase, TrainingArguments
from fate.ml.nn.model_zoo.agg_layer.fedpass.agg_layer import FedPassAggLayerGuest, FedPassAggLayerHost, get_model
from fate.ml.nn.model_zoo.agg_layer.sshe.agg_layer import SSHEAggLayerHost, SSHEAggLayerGuest
from fate.components.components.nn.runner.hetero_default_runner import DefaultRunner
import fate.components.components.hetero_nn

class TopSsl(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(t.zeros(1))

    def forward(self, x):
        return x

@dataclass
class SSLAggLayerArgument(Args):
    merge_type: Literal["sum", "concat", "ssl"] = "ssl"
    concat_dim = 1

    def to_dict(self):
        d = super().to_dict()
        d["agg_type"] = "ssl"
        return d

class AggLayerGuestSSL(AggLayerGuest):
    def __init__(self, merge_type: Literal["sum", "concat"] = "ssl", concat_dim=1, agg_type='std'):
        # super(AggLayerGuestSSL, self).__init__(merge_type, concat_dim=1)
        super(AggLayerGuest, self).__init__()
        self._host_input_caches = None
        self._merge_type = merge_type
        assert isinstance(concat_dim, int), "concat dim should be int"
        self.register_buffer("_concat_dim", t.LongTensor(concat_dim))

    def _forward(self, x_g: t.Tensor = None, x_h: List[t.Tensor] = None) -> t.Tensor:
        if x_g is None and x_h is None:
            raise ValueError("guest input and host inputs cannot be both None")

        if x_g is not None:
            x_g = x_g.to(self.device)
        if x_h is not None:
            x_h = [h.to(self.device) for h in x_h]

        can_cat = True
        if x_g is None:
            x_g = 0
            can_cat = False
        else:
            if self._model is not None:
                x_g = self._model(x_g)

        if x_h is None:
            ret = x_g
        else:
            if self._merge_type == "sum":
                for h_idx in range(len(x_h)):
                    x_g += x_h[h_idx]
                ret = x_g
            elif self._merge_type == "concat":
                # xg + x_h
                feat = [x_g] if can_cat else []
                feat.extend(x_h)
                ret = t.cat(feat, dim=1)# torch.Size([256, 8])  torch.Size([256, 16]) 也就是第二维度变成n方
            elif self._merge_type == "ssl":
                # mean and themselvs, x_g: [B, C], x_h: (n-1)*[B, C]
                _n = len(x_h) + 1
                _data = t.cat([x_g.unsqueeze(1)] + [_.unsqueeze(1) for _ in x_h], dim=1) # [B, N, C]
                _mean = _data.mean(dim=1, keepdim=False).detach() # [B, C]
                ret = [x_g, x_h, _mean]
            else:
                raise RuntimeError("unknown merge type")
        return ret

class AggLayerHostSSL(AggLayerHost):
    def __init__(self, merge_type, agg_type):
        self.merge_type = merge_type
        super(AggLayerHostSSL, self).__init__()

class HeteroNNModelGuestSSL(HeteroNNModelGuest):
    def __init__(
        self,
        top_model: t.nn.Module,
        bottom_model: t.nn.Module = None,
        agglayer_arg: Union[SSLAggLayerArgument, StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        top_arg: TopModelStrategyArguments = None,
        ctx: Context = None,
    ):
        super(HeteroNNModelGuestSSL, self).__init__(top_model, bottom_model, agglayer_arg, top_arg, ctx)
    def setup(
        self,
        ctx: Context = None,
        agglayer_arg: Union[SSLAggLayerArgument, StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        top_arg: TopModelStrategyArguments = None,
        bottom_arg=None,
    ):
        self._ctx = ctx

        if self._agg_layer is None:
            if agglayer_arg is None:
                self._agg_layer = AggLayerGuest()
            elif type(agglayer_arg) == SSLAggLayerArgument:
                self._agg_layer = AggLayerGuestSSL(**agglayer_arg.to_dict())
            elif type(agglayer_arg) == StdAggLayerArgument:
                self._agg_layer = AggLayerGuest(**agglayer_arg.to_dict())
            elif type(agglayer_arg) == FedPassArgument:
                self._agg_layer = FedPassAggLayerGuest(**agglayer_arg.to_dict())
            elif type(agglayer_arg) == SSHEArgument:
                self._agg_layer = SSHEAggLayerGuest(**agglayer_arg.to_dict())
                if self._bottom_model is None:
                    raise RuntimeError("A bottom model is needed when running a SSHE model")

        if self._top_add_model is None:
            if top_arg:
                logger.info("detect top model strategy")
                if top_arg.protect_strategy == "fedpass":
                    fedpass_arg = top_arg.fed_pass_arg
                    top_fedpass_model = get_model(**fedpass_arg.to_dict())
                    self._top_add_model = top_fedpass_model
                    self._top_model = t.nn.Sequential(self._top_model, top_fedpass_model)
                    if top_arg.add_output_layer == "sigmoid":
                        self._top_model.add_module("sigmoid", t.nn.Sigmoid())
                    elif top_arg.add_output_layer == "softmax":
                        self._top_model.add_module("softmax", t.nn.Softmax(dim=1))

        self._agg_layer.set_context(ctx)

    def forward(self, x=None):
        if self._agg_layer is None:
            self._auto_setup()

        if self.device is None:
            self.device = self.get_device(self._top_model)
            self._agg_layer.set_device(self.device)
            if isinstance(self._agg_layer, SSHEAggLayerHost):
                if self.device.type != "cpu":
                    raise ValueError("SSHEAggLayerGuest is not supported on GPU")

        if self._bottom_model is None:
            b_out = None
        else:
            b_out = self._bottom_model(x)
            # bottom layer
            if not self._guest_direct_backward:
                self._bottom_fw = b_out

        # hetero layer
        if not self._guest_direct_backward:
            agg_out = self._agg_layer.forward(b_out)
            self._agg_fw_rg = agg_out.requires_grad_(True)
            # top layer
            top_out = self._top_model(self._agg_fw_rg)
        else:
            # top_out = self._top_model(self._agg_layer(b_out))
            x_g, x_h, _mean = self._agg_layer.forward(b_out) # [x_g, x_h, _mean]
            top_out = self._top_model(x_g)
            top_out_h = [self._top_model(x) for x in x_h]

        return [top_out, top_out_h, x_g, x_h, _mean]
    
class HeteroNNModelHostSSL(HeteroNNModelHost):
    def __init__(
        self,
        bottom_model: t.nn.Module,
        agglayer_arg: Union[SSLAggLayerArgument, StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        ctx: Context = None,
    ):
        super().__init__(bottom_model, agglayer_arg, ctx)
    
    def setup(
        self,
        ctx: Context = None,
        agglayer_arg: Union[StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        bottom_arg=None,
    ):
        self._ctx = ctx

        if self._agg_layer is None:
            if agglayer_arg is None:
                self._agg_layer = AggLayerHost()
            elif type(agglayer_arg) == SSLAggLayerArgument:
                self._agg_layer = AggLayerHostSSL(**agglayer_arg.to_dict())
            elif type(agglayer_arg) == StdAggLayerArgument:
                self._agg_layer = AggLayerHost()  # no parameters are needed
            elif type(agglayer_arg) == FedPassArgument:
                self._agg_layer = FedPassAggLayerHost(**agglayer_arg.to_dict())
            elif isinstance(agglayer_arg, SSHEArgument):
                self._agg_layer = SSHEAggLayerHost(**agglayer_arg.to_dict())

        self._agg_layer.set_context(ctx)

class HeteroNNTrainerGuestSSL(HeteroNNTrainerGuest):
    def __init__(
        self,
        ctx: Context,
        model: HeteroNNModelGuestSSL,
        training_args: TrainingArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: nn.Module = None,
        loss_fn_ssl: nn.Module = None,
        optimizer=None,
        data_collator: Callable = None,
        scheduler=None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        super().__init__(
            ctx=ctx,
            model=model,
            training_args=training_args,
            train_set=train_set,
            val_set=val_set,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_collator=data_collator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        self.loss_fn_ssl = loss_fn_ssl

    def compute_loss(self, model, inputs, **kwargs):
        # mean lossn len(inputs)=2
        feats, labels = inputs
        # feats: [B, C]. labels: [B, 1].
        pred, pred_h, x_g, x_h, _mean = model(feats)
        loss = self.loss_fn_ssl(x_g, _mean)
        for _x_h in x_h:
            loss += self.loss_fn_ssl(_x_h, _mean)
        # loss = loss / (len(x_h)+1) + self.loss_func(pred, labels)
        loss += self.loss_func(pred, labels)
        for _pred_h in pred_h:
            loss += self.loss_func(_pred_h, labels)
        loss = loss / (len(x_h)+1)
        return loss

######################################################
# 自监督训练流程覆写
######################################################
def train(ctx: Context, 
          dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          loss_fn_ssl = None, 
          args: TrainingArguments = None, 
          ):

    if ctx.is_on_guest:
        trainer = HeteroNNTrainerGuestSSL(ctx=ctx,
                                       model=model,
                                       train_set=dataset,
                                       optimizer=optimizer,
                                       loss_fn=loss_func,
                                       loss_fn_ssl=loss_fn_ssl,
                                       training_args=args
                                       )
    else:
        trainer = HeteroNNTrainerHost(ctx=ctx,
                                      model=model,
                                      train_set=dataset,
                                      optimizer=optimizer,
                                      training_args=args
                                    )

    trainer.train()
    return trainer

def predict(trainer, dataset):
    return trainer.predict(dataset)

from model import MLP2, projection_mlp
# model = nn.Sequential(
#     MLP2(5, [32, 32]),
#     projection_mlp(32, 32, 32, 3)
# )
_dim = 256
def get_setting(ctx):
    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        # ds.load("./breast_hetero_guest.csv")
        ds.load("./guest.csv")

        # bottom_model = t.nn.Sequential(
        #     t.nn.Linear(5, 32),
        #     t.nn.ReLU(),
        # )
        bottom_model = nn.Sequential(
                            MLP2(5, [_dim, _dim]),
                            projection_mlp(_dim, _dim, _dim, 3)
                        )
        top_model = t.nn.Sequential(
            t.nn.Linear(_dim, 1),
            t.nn.Sigmoid()
        )
        # top_model = TopSsl()
        model = HeteroNNModelGuestSSL(
            top_model=top_model,
            bottom_model=bottom_model,
            agglayer_arg=SSLAggLayerArgument()
            # agglayer_arg=SSHEArgument(
            #     guest_in_features=8,
            #     host_in_features=8,
            #     out_features=8,
            #     layer_lr=0.01
            # )
        )#.load_state_dict

        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = t.nn.BCELoss()
        loss_fn_ssl = NegCosineSimilarityLoss()
        # loss_fn_ssl = NTXentLoss(memory_bank_size=1024)

    else:
        ds = TableDataset(to_tensor=True)
        # ds.load("./breast_hetero_host.csv")
        ds.load("./host.csv")
        # bottom_model = t.nn.Sequential(
        #     t.nn.Linear(5, 32),
        #     t.nn.ReLU(),
        # )
        bottom_model = nn.Sequential(
                            MLP2(5, [_dim, _dim]),
                            projection_mlp(_dim, _dim, _dim, 3)
                        )
        model = HeteroNNModelHostSSL(
            bottom_model=bottom_model,
            agglayer_arg=SSLAggLayerArgument()
            # agglayer_arg=SSHEArgument(
            #     guest_in_features=8,
            #     host_in_features=8,
            #     out_features=8,
            #     layer_lr=0.01
            # )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = None
        loss_fn_ssl = None

    args = TrainingArguments(
        num_train_epochs=100,
        per_device_train_batch_size=256
    )

    return ds, model, optimizer, loss, loss_fn_ssl, args

def run(ctx):
    ds, model, optimizer, loss, loss_fn_ssl, args = get_setting(ctx)
    trainer = train(ctx, ds, model, optimizer, loss, loss_fn_ssl, args)
    # 训练并保存各方子监督预训练结果
    t.save(trainer.model.state_dict(), 'guest-rank('+str(ctx.rank)+').pt' if ctx.is_on_guest else 'host-rank('+str(ctx.rank)+').pt')
    pred = predict(trainer, ds)
    if ctx.is_on_guest:
        # print("pred:", pred)
        # compute auc here
        from sklearn.metrics import roc_auc_score
        # print('auc is')
        # print(roc_auc_score(pred.label_ids, pred.predictions))
        # top_out, x_g, x_h, _mean
        top_out, top_out_h, x_g, x_h, _mean = pred.predictions
        print('auc is', roc_auc_score(pred.label_ids, top_out))

# python stage1.py --parties guest:9999 host:10000 --log_level INFO
# python stage1.py --parties guest:9999 host:10000 host:10001 --log_level INFO
if __name__ == '__main__':
    from fate.arch.launchers.multiprocess_launcher import launch
    launch(run)
