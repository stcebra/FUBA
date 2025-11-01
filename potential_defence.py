import torch
import pickle
import torch.nn.functional as F


def federated_averaging_enforce_training(local_net_dicts, clear_dataset, device='cuda:0', epochs=5, lr=0.001, unlearn_global_path='checkpoints_ori/global_net_model_test_round_100.pkl', mu=0.01,config=None):
    """
    Args:
        global_net_dict: dict, the state_dict of global model before averaging
        local_net_dicts: list of state_dicts from clients
        clear_dataset: torch.utils.data.Dataset, dataset for additional training
        device: device to use ('cuda' or 'cpu')
        epochs: number of epochs for enforced training
        lr: learning rate for enforced training
    
    Returns:
        dict: updated global model's state_dict after averaging and enforced training
    """
    local_nets = local_net_dicts[:]
    i = 0
    while i < len(local_nets):
        if local_nets[i] is None:
            local_nets.pop(i)
        else:
            i += 1

    if not local_nets:
        return None

    # Step 1: Federated Averaging
    averaged_state = {}
    for k in local_nets[0].keys():
        local_updates = [local_net[k].float() for local_net in local_nets]
        avg_update = torch.stack(local_updates, dim=0).mean(dim=0)
        averaged_state[k] = avg_update

    # Step 2: Load unlearned global model with torch.load
    with open(unlearn_global_path, 'rb') as f:
        unlearned_state_dict = pickle.load(f)
    unlearned_state_dict = {k: v.to(device) for k, v in unlearned_state_dict.items()}
    # Step 3: Enforced Training on Clear Dataset
    model = config.creat_cls_net().to(device)
    model.load_state_dict(averaged_state)

    dataloader = torch.utils.data.DataLoader(clear_dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            #to do add regulation 
            loss = criterion(output, y)

            # Add regularization
            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in unlearned_state_dict:
                    prox_term += ((param - unlearned_state_dict[name].to(device)) ** 2).sum()
            loss += (mu / 2) * prox_term
            loss.backward()
            optimizer.step()

    return model.state_dict()



def federated_averaging_topk(local_net_dicts,device='cuda:0',unlearned_state_dict=None,topk=4):
    """
    Args:
        global_net_dict: dict, the state_dict of global model before averaging
        local_net_dicts: list of state_dicts from clients
        clear_dataset: torch.utils.data.Dataset, dataset for additional training
        device: device to use ('cuda' or 'cpu')
        epochs: number of epochs for enforced training
        lr: learning rate for enforced training
    
    Returns:
        dict: updated global model's state_dict after averaging and enforced training
    """
    local_nets = local_net_dicts[:]
    i = 0
    while i < len(local_nets):
        if local_nets[i] is None:
            local_nets.pop(i)
        else:
            i += 1

    if not local_nets:
        return None

    # Step 2: Load unlearned global model
    unlearned_state_dict = {k: v.to(device) for k, v in unlearned_state_dict.items()}
    # Flatten all parameters of unlearned model
    unlearn_vec = torch.cat([v.flatten() for v in unlearned_state_dict.values()]).to(device)

    # Compute cosine similarity for each client model
    sims = []
    for i, net in enumerate(local_nets):
        model_vec = torch.cat([net[k].flatten().to(device) for k in net])
        cos_sim = F.cosine_similarity(model_vec.unsqueeze(0), unlearn_vec.unsqueeze(0)).item()
        sims.append((cos_sim, i))

    # Select top-k most similar clients
    sims.sort(reverse=True)
    topk_indices = [idx for _, idx in sims[:topk]]
    selected_nets = [local_nets[i] for i in topk_indices]

    # FedAvg on selected models
    averaged_state = {}
    for k in selected_nets[0].keys():
        local_updates = [net[k].float() for net in selected_nets]
        avg_update = torch.stack(local_updates, dim=0).mean(dim=0)
        averaged_state[k] = avg_update

    return averaged_state
