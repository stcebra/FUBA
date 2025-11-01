# aggregation.py
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from argparse import Namespace

def run_aggregation(global_model, state_dict_list, agent_data_sizes, config,aggr_method="alignins", device=None, args=None):
    if device is None:
        device = next(global_model.parameters()).device

    args = Namespace(
        aggr=aggr_method,
        server_lr=1.0,
        dense_ratio=0.25,
        num_corrupt=0,
        sparsity=0.3,
        theta=8,
        lambda_s=1.0,
        lambda_c=1.0,
        device=device
    )

    flat_global = parameters_to_vector(global_model.state_dict().values()).detach()
    client_updates = {
        idx: parameters_to_vector(sd.values()).to(flat_global.device) - flat_global
        for idx, sd in enumerate(state_dict_list) if sd is not None
    }

    aggregator = config.Aggregation(agent_data_sizes, flat_global.numel(), args)
    return aggregator.aggregate_updates(global_model, client_updates)

def masa_aggregate(global_model, local_dicts, data_sizes, args,config):
    ns = Namespace(
        aggr='masa',
        server_lr=args.server_lr,
        dense_ratio=args.dense_ratio,
        num_corrupt=args.num_corrupt,
        sparsity=getattr(args, 'sparsity', 0.3),
        theta=args.theta,
        lambda_s=getattr(args, 'lambda_s', 1.0),
        lambda_c=getattr(args, 'lambda_c', 1.0),
        device=args.device
    )
    flat_global = parameters_to_vector(global_model.state_dict().values()).detach()
    aggr = config.Aggregation(data_sizes, flat_global.numel(), ns)
    updates_dict, _ = aggr.aggregate_updates(global_model, 
                                             {i: parameters_to_vector(sd.values())-flat_global
                                              for i, sd in enumerate(local_dicts) if sd is not None})
    delta = torch.cat([updates_dict[name].flatten() for name in updates_dict])
    vector_to_parameters(flat_global + delta, global_model.parameters())
    return global_model.state_dict()
