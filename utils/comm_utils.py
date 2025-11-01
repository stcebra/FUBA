import torch
import numpy as np
import os
import pickle
import random
from torch.utils.data import random_split, TensorDataset
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from collections import OrderedDict


def l2_distance(model, reference_model):
    """Compute L2 distance (4th power) between two models' parameters."""
    distance = 0.0
    for param, ref_param in zip(model.parameters(), reference_model.parameters()):
        distance += torch.sum((param - ref_param) ** 4) * 10000
    return distance


def l1_distance_dict(dict1, dict2):
    """Compute L1 distance between two state_dicts."""
    distance = 0
    for key in dict1.keys():
        distance += (dict1[key] - dict2[key]).abs().sum()
    return distance


def attack(model, loader, atkmodel, ATK_EPS, NOISE_THREAD, target_label,
           device, with_sample=False,**kwargs):
    """Perform a backdoor attack and measure attack success rate (ASR)."""
    model.eval()
    backdoor_correct = 0
    total = 0
    i = 0
    if with_sample:
        samples = {}

    with torch.no_grad():
        for data in loader:
            i += 1
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            # Save clean sample for visualization
            if with_sample and i < 4:
                samples[i] = images[0]

            original_labels = labels.clone()
            noise = atkmodel(images) * ATK_EPS
            noise = torch.clamp(noise, -NOISE_THREAD, NOISE_THREAD)
            images = torch.clamp(images + noise, -1.0, 1.0)

            # Save backdoored sample for visualization
            if with_sample and i < 4:
                samples[i + 0.5] = images[0]

            labels.fill_(target_label)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += (original_labels != target_label).sum().item()
            backdoor_correct += ((predicted == labels) & (original_labels != target_label)).sum().item()

    if with_sample:
        return 100 * backdoor_correct / total, samples

    return 100 * backdoor_correct / total


def attack_dba(model, loader, target_label, device, backdoor_func,**kwargs):
    """Perform DBA (distributed backdoor attack) with a custom backdoor function."""
    model.eval()
    backdoor_correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            original_labels = labels.clone()
            images = backdoor_func(images)
            labels.fill_(target_label)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (original_labels != target_label).sum().item()
            backdoor_correct += ((predicted == labels) & (original_labels != target_label)).sum().item()
    return 100 * backdoor_correct / total


def test(model, loader, device):
    """Evaluate model accuracy on a test dataset."""
    model.eval()
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data in loader:
            i += 1
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total*100


def get_backdoored_dataset(model, loader, atkmodel,
                           target_label, ATK_EPS, NOISE_THREAD,
                           num_samples=200,
                           device=None):
    """Generate a dataset of backdoored samples for evaluation or training."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    backdoored_imgs = []
    backdoored_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Use only non-target class samples
            mask = (labels != target_label)
            if mask.sum().item() == 0:
                continue

            imgs = images[mask]

            noise = atkmodel(imgs) * ATK_EPS
            noise = torch.clamp(noise, -NOISE_THREAD, NOISE_THREAD)

            bd_imgs = torch.clamp(imgs + noise, -1.0, 1.0)

            backdoored_imgs.append(bd_imgs.cpu())
            backdoored_labels.append(
                torch.full((bd_imgs.size(0),), target_label, dtype=torch.long)
            )

            total = sum(x.size(0) for x in backdoored_imgs)
            if total >= num_samples:
                break

    all_imgs = torch.cat(backdoored_imgs, dim=0)[:num_samples]
    all_labels = torch.cat(backdoored_labels, dim=0)[:num_samples]

    return TensorDataset(all_imgs, all_labels)


def get_clean_dataset(loader, num_samples=200, device=None):
    """Sample a clean dataset subset from a DataLoader."""
    clean_imgs = []
    clean_labels = []
    collected = 0

    for images, labels in loader:
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
            images = images.cpu()
            labels = labels.cpu()

        clean_imgs.append(images)
        clean_labels.append(labels)
        collected += images.size(0)

        if collected >= num_samples:
            break

    all_imgs = torch.cat(clean_imgs, dim=0)[:num_samples]
    all_labels = torch.cat(clean_labels, dim=0)[:num_samples]

    return TensorDataset(all_imgs, all_labels)

def state_dict_to_cpu(state_dict):
    """
    Ensure all tensors in a state_dict are moved to CPU.
    
    Args:
        state_dict (dict or OrderedDict): model.state_dict() like object
    
    Returns:
        OrderedDict: new state_dict on CPU
    """
    cpu_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_state_dict[k] = v.detach().cpu()
        else:
            # just in case there are non-Tensor values
            cpu_state_dict[k] = v
    return cpu_state_dict

def save_models(global_model_dict, local_model_dicts, backdoor_model, r, name):
    """Save global, local, and backdoor model states to disk."""
    os.makedirs("checkpoints", exist_ok=True)

    with open(f"checkpoints/global_net_model_{name}_round_{r}.pkl", "wb") as f:
        pickle.dump(state_dict_to_cpu(global_model_dict), f)

    
    if local_model_dicts is not None:
        for i in range(len(local_model_dicts)):
            local_model_dicts[i] = state_dict_to_cpu(local_model_dicts[i])
        with open(f"checkpoints/local_net_model_{name}_round_{r}.pkl", "wb") as f:
            pickle.dump(local_model_dicts, f)

    if backdoor_model is not None:
        with open(f"checkpoints/backdoor_net_model_{name}_round_{r}.pkl", "wb") as f:
            pickle.dump(state_dict_to_cpu(backdoor_model.state_dict()), f)


def split_non_iid_concept(trainset, trainset_split, shift_ratio=0.4, seed=123):
    """Apply concept shift to part of the dataset to simulate non-IID distribution."""
    assert 0 < shift_ratio <= 1.0
    num_clients = len(trainset_split)
    num_shift_clients = int(num_clients * shift_ratio)
    type_split_N = num_shift_clients // 2

    random.seed(seed)
    random_clients = list(range(num_clients))
    random.shuffle(random_clients)

    C = 10  # number of classes

    for i in range(5, num_shift_clients):
        sampled_client = random_clients[i]
        target_dataset = trainset_split[sampled_client]

        for idx in target_dataset.indices:
            original_label = trainset.targets[idx].item()
            if i < type_split_N:
                new_label = C - 1 - original_label  # flip labels
            else:
                new_label = (original_label + 1) % C  # shift labels
            trainset.targets[idx] = new_label


def lognormal_split(trainset, num_clients, mean=2.0, sigma=1.0, seed=0):
    """Split dataset into clients using lognormal distribution."""
    np.random.seed(seed)

    total_len = len(trainset)
    raw_sizes = np.random.lognormal(mean=mean, sigma=sigma, size=num_clients)
    proportions = raw_sizes / raw_sizes.sum()

    lengths = np.round(proportions * total_len).astype(int)
    print(lengths)

    diff = total_len - lengths.sum()
    lengths[np.argmax(lengths)] += diff

    subsets = random_split(trainset, lengths.tolist(), generator=torch.Generator().manual_seed(seed))
    return subsets


def split_non_iid_dirichlet(dataset, DATASET, num_clients=5, alpha=0.5,
                            min_samples_per_client=100, max_retries=1000, seed=1223):
    """Split dataset into clients using Dirichlet distribution for non-IID simulation."""
    if DATASET == "Imagenet":
        num_classes = 1000
        raise ValueError("TODO")
    elif DATASET == "mnist":
        num_classes = 10
        y = dataset.targets
    elif DATASET == "cifar10":
        num_classes = 10
        y = dataset.targets

    for attempt in range(max_retries):
        class_indices = [np.where(y == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]

        for cls_id, idxs in enumerate(class_indices):
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split = np.split(idxs, proportions)
            for i, idx in enumerate(split):
                client_indices[i].extend(idx)

        client_lens = [len(idxs) for idxs in client_indices]
        break

    subsets = [torch.utils.data.Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return subsets


def find_max_idx_within_limit(lst, limit):
    """Find index of the maximum value within a limit."""
    max_value = -float('inf')
    max_idx = -1
    for idx, value in enumerate(lst):
        if value <= limit and value > max_value:
            max_value = value
            max_idx = idx
    return max_idx


def federated_averaging_net(local_nets):
    """Perform federated averaging over a list of local models."""
    local_nets = local_nets[:]
    i = 0
    while i < len(local_nets):
        if local_nets[i] is None:
            local_nets.pop(i)
        else:
            i += 1

    d = {}
    if not local_nets:
        return None

    for k in local_nets[0].keys():
        local_updates = [local_net[k].float() for local_net in local_nets]
        avg_update = torch.stack(local_updates, 0).mean(0)
        d[k] = avg_update
    return d


def geometric_median(gradients):
    """Compute geometric median of a list of gradients."""
    def aggregate_loss(grad):
        return sum(euclidean(grad, g) for g in gradients)

    result = minimize(aggregate_loss, np.mean(gradients, axis=0))
    return result.x


def compute_parameter_ratio_difference(model_a, model_b, min_scale=0.1, max_scale=1.0):
    """Compute ratio differences between two model parameter sets."""
    param_ratios = {}
    max_diff = 0.0
    min_diff = float('inf')

    for (name_a, param_a), (name_b, param_b) in zip(model_a.items(), model_b.items()):
        diff = torch.abs(param_b - param_a)
        param_ratios[name_a] = diff
        max_diff = max(max_diff, diff.max().item())
        min_diff = min(min_diff, diff.min().item())

    for name in param_ratios:
        param_ratios[name] = min_scale + (max_scale - min_scale) * (param_ratios[name] - min_diff) / (max_diff - min_diff)

    return param_ratios


def adjust_updates_based_on_ratio(model, param_ratios):
    """Adjust gradients of model parameters according to ratio dictionary."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.data.mul_(param_ratios[name])


def dict_to_cpu(state_dict):
    """Move a state_dict to CPU."""
    return {k: v.cpu() for k, v in state_dict.items()}


def dict_to_device(state_dict, device):
    """Move a state_dict to a specific device."""
    return {k: v.to(device) for k, v in state_dict.items()}
