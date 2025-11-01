import torch
import copy
import random
import datetime
from argparse import Namespace
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from torch.utils.data import Subset

from utils.comm_utils import federated_averaging_net, geometric_median
from .simple_defence_method import federated_averaging_median, federated_averaging_multi_krum
from .aggregtion import run_aggregation
from potential_defence import federated_averaging_enforce_training, federated_averaging_topk
from config import Config

# Global aggregator (used in IBMFL)
aggregator = None


def defence(config, global_model_dict, local_model_dicts, backdoor_model_dicts, trainset, testset):
    """
    Perform defense aggregation strategy based on configuration flags.

    Args:
        config (Config): Experiment configuration object.
        local_model_dicts (list[dict]): State dicts of local client models.
        backdoor_model_dicts (list[dict]): State dicts of local backdoor models.
        trainset (Dataset): Training dataset.
        testset (Dataset): Testing dataset.

    Returns:
        tuple: (global_model_dict, global_backdoor_model_dict)
    """

    # ---- Geometric Median Defence ----
    if config.apply_geometric_median:
        aggregated_model = geometric_median(local_model_dicts)

    # ---- FLAME Defence ----
    if config.apply_flame:
        args = Namespace(
            frac=0.4,       # fraction of participating clients
            num_users=20,   # number of users
            malicious=0.2,  # proportion of malicious clients
            noise=0.001,    # Gaussian noise factor
            wrong_mal=0,
            right_ben=0,
            turn=0
        )
        updates = []
        for model_dict in local_model_dicts:
            if model_dict is not None:
                updates.append(config.get_update(model_dict, global_model_dict))

        local_model_dicts_no_None = [i for i in local_model_dicts if i is not None]
        global_model_dict = config.flame(local_model_dicts_no_None, updates, global_model_dict, args)

    # ---- Multi-Krum Defence ----
    elif config.apply_multy_krum:
        global_model_dict = federated_averaging_multi_krum(global_model_dict, local_model_dicts)

    # ---- Median Defence ----
    elif config.apply_median:
        global_model_dict = federated_averaging_median(global_model_dict, local_model_dicts)

    # ---- AlignIns Defence ----
    elif config.apply_alignins:
        global_model = config.creat_cls_net()
        updates_dict, neurotoxin_mask = run_aggregation(
            global_model,
            local_model_dicts,
            [10000 * 20],
            config
        )
        global_model_dict = global_model.state_dict()

    # ---- Indicator Defence ----
    elif config.apply_indicator:
        sample_data, _ = trainset[1]
        server = config.IndicatorServer(
            params=config.indicator_param,
            current_time=datetime.datetime.now().strftime("%b.%d_%H.%M.%S"),
            train_dataset=trainset,
            open_set=config.indicator_loader.ood_data,
            blend_pattern=(torch.rand(sample_data.shape) - 0.5) * 2,
            edge_case_train=config.indicator_loader.edge_poison_train,
            edge_case_test=config.indicator_loader.edge_poison_test
        )
        global_model = config.creat_cls_net().to(config.device)
        global_model.load_state_dict(global_model_dict)
        server.global_model = global_model

        # Identify benign clients
        benign_client_indices = server._indicator(
            local_model_state_dict=local_model_dicts,
            wm_data=config.indicator_loader.ood_data
        )

        # Keep only benign client updates
        filtered_state_dicts = [
            local_model_dicts[i] for i in benign_client_indices
        ]
        if not filtered_state_dicts:
            global_model_dict = federated_averaging_net(local_model_dicts)
        else:
            global_model_dict = federated_averaging_net(filtered_state_dicts)
        if global_model_dict is None:
            global_model_dict = federated_averaging_net(local_model_dicts)

    # ---- IBMFL Defence ----
    elif config.apply_ibmfl and config.round > 5:
        gn = config.creat_cls_net().state_dict()
        net = config.creat_cls_net().to(config.device)
        net.load_state_dict(global_model_dict)
        if config.aggrator is None:
            config.aggrator = config.Aggregation(
                [2000] * config.num_clients,
                len(parameters_to_vector([gn[name] for name in gn])),
                None,
                config.IBMFL_args,
                None,
                testset
            )
        local_params = {
            i: parameters_to_vector([local_model_dicts[i][name] for name in global_model_dict]).detach().to(config.device)
            - parameters_to_vector([global_model_dict[name] for name in global_model_dict]).detach().to(config.device)
            for i in range(len(local_model_dicts)) if local_model_dicts[i] is not None
        }
        global_update, neurotoxin_mask = config.aggrator.aggregate_updates(net, local_params)
        
        cur_global_dict = net.state_dict()

        new_global_dict = {}
        for k in cur_global_dict.keys():
            new_global_dict[k] = cur_global_dict[k].to(config.device) + global_update[k].to(config.device)

        net.load_state_dict(new_global_dict)
        global_model_dict = net.state_dict()

    # ---- Default Federated Averaging ----
    elif config.apply_fldetector:
        from .FLDetector.train_mnist import detection
        from sklearn.cluster import KMeans

        ref_keys = list(global_model_dict.keys())

        def _flatten_state_dict(sd, keys):
            vecs = []
            for k in keys:
                t = sd[k].to(config.device)
                vecs.append(t.reshape(-1))
            return torch.cat(vecs, dim=0)

        updates = []
        kept_indices = [] 
        gvec = _flatten_state_dict(global_model_dict, ref_keys)

        for i, local_sd in enumerate(local_model_dicts):
            if local_sd is None:
                continue
            lvec = _flatten_state_dict(local_sd, ref_keys)
            updates.append(lvec - gvec)      
            kept_indices.append(i)

        if len(updates) == 0:
            global_model_dict = federated_averaging_net(local_model_dicts)
        elif len(updates) == 1:
            global_model_dict = local_model_dicts[kept_indices[0]]
        else:
            U = torch.stack(updates, dim=0)                 
            center = U.median(dim=0).values                 
            scores = torch.norm(U - center, dim=1) / (U.size(1) ** 0.5)
            scores_np = scores.detach().cpu().numpy()

            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
            labels = kmeans.fit_predict(scores_np.reshape(-1, 1))
            mean0 = scores_np[labels == 0].mean()
            mean1 = scores_np[labels == 1].mean()
            benign_label = 0 if mean0 < mean1 else 1

            benign_mask = (labels == benign_label)
            benign_client_indices = [kept_indices[i] for i, flag in enumerate(benign_mask) if flag]

            if len(benign_client_indices) == 0:
                filtered_state_dicts = [local_model_dicts[i] for i in kept_indices]
            else:
                filtered_state_dicts = [local_model_dicts[i] for i in benign_client_indices]

            global_model_dict = federated_averaging_net(filtered_state_dicts)
    else:
        global_model_dict = federated_averaging_net(local_model_dicts)

    # Always aggregate backdoor models using FedAvg
    global_backdoor_model_dict = federated_averaging_net(backdoor_model_dicts)

    return global_model_dict, global_backdoor_model_dict
