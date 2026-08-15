import torch as t
from fate.arch import Context
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.model_zoo.hetero_nn_model import SSHEArgument, FedPassArgument, TopModelStrategyArguments



def train(ctx: Context, 
          dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          args: TrainingArguments = None, 
          ):

    if ctx.is_on_guest:
        trainer = HeteroNNTrainerGuest(ctx=ctx,
                                       model=model,
                                       train_set=dataset,
                                       optimizer=optimizer,
                                       loss_fn=loss_func,
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
_dim = 256

def get_setting(ctx):

    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./guest.csv")

        bottom_model = t.nn.Sequential(
                            MLP2(5, [_dim, _dim]),
                            projection_mlp(_dim, _dim, _dim, 3)
                        )
        bottom_model.load_state_dict(t.load('stage2/premodels/model_encoder_local_bottom-mlp2-0.pth'))
        top_model = t.nn.Sequential(
            t.nn.Linear(_dim, 1),
            t.nn.Sigmoid()
        )
        bottom_model.load_state_dict(t.load('stage2/premodels/model_encoder_local_top-mlp2-0.pth'))
        model = HeteroNNModelGuest(
            top_model=top_model,
            bottom_model=bottom_model,
            agglayer_arg=SSHEArgument(
                guest_in_features=8,
                host_in_features=8,
                out_features=8,
                layer_lr=0.01
            )
        )

        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = t.nn.BCELoss()

    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./host.csv")
        bottom_model = t.nn.Sequential(
                            MLP2(5, [_dim, _dim]),
                            projection_mlp(_dim, _dim, _dim, 3)
                        )
        bottom_model.load_state_dict(t.load('stage2/premodels/model_encoder_local_bottom-mlp2-1.pth'))

        model = HeteroNNModelHost(
            bottom_model=bottom_model,
            agglayer_arg=SSHEArgument(
                guest_in_features=8,
                host_in_features=8,
                out_features=8,
                layer_lr=0.01
            )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = None

    args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=256
    )

    return ds, model, optimizer, loss, args


def run(ctx):
    ds, model, optimizer, loss, args = get_setting(ctx)
    trainer = train(ctx, ds, model, optimizer, loss, args)
    pred = predict(trainer, ds)
    if ctx.is_on_guest:
        # print("pred:", pred)
        # compute auc here
        from sklearn.metrics import roc_auc_score
        print('auc is')
        print(roc_auc_score(pred.label_ids, pred.predictions))


if __name__ == '__main__':
    from fate.arch.launchers.multiprocess_launcher import launch
    launch(run)
    # python stage3.py --parties guest:9999 host:10000 --log_level INFO