import torch
import copy


def euclidean_distance(tensor1, tensor2):
    """Compute Euclidean (L2) distance between two tensors."""
    return torch.norm(tensor1 - tensor2, p=2)


def krum_selection(local_updates, f=10):
    """
    Perform Krum selection to choose the most representative client update.

    Args:
        local_updates (list[Tensor]): List of local model updates (one per client).
        f (int): Maximum number of Byzantine (malicious) clients tolerated.

    Returns:
        Tensor: The selected client update.
    """
    num_clients = len(local_updates)
    scores = []

    # Compute score for each client update
    for i in range(num_clients):
        distances = []
        for j in range(num_clients):
            if i != j:
                distances.append(euclidean_distance(local_updates[i], local_updates[j]).item())

        # Sort distances to other updates and take closest num_clients - f - 2
        distances.sort()
        scores.append((i, sum(distances[:num_clients - f - 2])))

    # Select the client with the smallest score
    scores.sort(key=lambda x: x[1])
    selected_idx = scores[0][0]
    return local_updates[selected_idx]


def federated_averaging_krum(global_net_dict, local_net_dicts, f=10):
    """
    Federated averaging with Krum selection strategy.

    Args:
        global_net_dict (dict): Global model state_dict.
        local_net_dicts (list[dict]): List of local model state_dicts.
        f (int): Maximum number of Byzantine (malicious) clients tolerated.

    Returns:
        dict: Updated global model parameters.
    """
    local_net_dicts = local_net_dicts[:]
    i = 0
    while i < len(local_net_dicts):
        if local_net_dicts[i] is None:
            local_net_dicts.pop(i)
        else:
            i += 1

    if len(local_net_dicts) == 0:
        raise ValueError("No local updates available for aggregation.")

    global_dict = copy.deepcopy(global_net_dict)
    global_dict_prev = global_net_dict

    for k in global_dict.keys():
        # Compute update vectors relative to previous global model
        local_updates = [
            (local_net_dict[k].float() - global_dict_prev[k].float())
            for local_net_dict in local_net_dicts
        ]

        # Select best update using Krum
        best_update = krum_selection(local_updates, f)
        global_dict[k] = global_dict_prev[k] + best_update

    return global_dict


def multi_krum_selection(local_updates, f=2, m=None):
    """
    Perform Multi-Krum selection (averaging multiple selected updates).

    Args:
        local_updates (list[Tensor]): List of local model updates.
        f (int): Maximum number of Byzantine (malicious) clients tolerated.
        m (int): Number of updates to average. Default = num_clients - f - 2.

    Returns:
        Tensor: Averaged update vector from selected clients.
    """
    num_clients = len(local_updates)
    if m is None:
        m = num_clients - f - 2

    scores = []

    # Score each update based on distances to others
    for i in range(num_clients):
        distances = []
        for j in range(num_clients):
            if i != j:
                distances.append((j, euclidean_distance(local_updates[i], local_updates[j]).item()))

        # Sort by distance and take top-m closest
        distances.sort(key=lambda x: x[1])
        total_distance = sum([dist[1] for dist in distances[:m]])
        scores.append((i, total_distance))

    # Select m updates with the smallest scores
    scores.sort(key=lambda x: x[1])
    selected_indices = [scores[i][0] for i in range(m)]
    # Average selected updates
    selected_updates = [local_updates[idx] for idx in selected_indices]
    averaged_update = sum(selected_updates) / len(selected_updates)

    return averaged_update


def federated_averaging_multi_krum(global_net_dict, local_net_dicts, f=10):
    """
    Federated averaging using Multi-Krum selection.

    Args:
        global_net_dict (dict): Global model state_dict.
        local_net_dicts (list[dict]): List of local model state_dicts.
        f (int): Maximum number of Byzantine (malicious) clients tolerated.

    Returns:
        dict: Updated global model parameters.
    """
    local_net_dicts = [net_dict for net_dict in local_net_dicts if net_dict is not None]

    if len(local_net_dicts) == 0:
        raise ValueError("No local updates available for aggregation.")

    global_dict = copy.deepcopy(global_net_dict)
    global_dict_prev = global_net_dict

    for k in global_dict.keys():
        local_updates = [
            (local_net_dict[k].float() - global_dict_prev[k].float())
            for local_net_dict in local_net_dicts
        ]
        best_update = multi_krum_selection(local_updates, f, 4)
        global_dict[k] = global_dict_prev[k] + best_update

    return global_dict


def federated_averaging_median(global_net_dict, local_net_dicts):
    """
    Federated averaging using coordinate-wise median aggregation.

    Args:
        global_net_dict (dict): Global model state_dict.
        local_net_dicts (list[dict]): List of local model state_dicts.

    Returns:
        dict: Updated global model parameters.
    """
    local_net_dicts = local_net_dicts[:]
    i = 0
    while i < len(local_net_dicts):
        if local_net_dicts[i] is None:
            local_net_dicts.pop(i)
        else:
            i += 1

    if len(local_net_dicts) == 0:
        raise ValueError("No local updates available for aggregation.")

    global_dict = global_net_dict.copy()
    global_dict_prev = global_net_dict.copy()

    for k in global_dict.keys():
        local_updates = [
            (local_net_dict[k].float() - global_dict_prev[k].float())
            for local_net_dict in local_net_dicts
        ]

        # Take coordinate-wise median
        median_update = torch.median(torch.stack(local_updates, dim=0), dim=0)[0]
        global_dict[k] = global_dict_prev[k] + median_update

    return global_dict
