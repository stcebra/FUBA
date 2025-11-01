import torch
import copy
from utils.comm_utils import attack
from utils.comm_utils import l2_distance, attack, test,get_backdoored_dataset ,get_clean_dataset, save_models, split_non_iid_dirichlet,find_max_idx_within_limit,federated_averaging_net,geometric_median,compute_parameter_ratio_difference,adjust_updates_based_on_ratio,dict_to_cpu,dict_to_device,lognormal_split,split_non_iid_concept
from config import Config
import random
import torch.nn as nn
import torch.optim as optim
from model import UNet,MNISTAutoencoder
from dataset import MyImagenet
from torch.utils.data import Subset
from potential_defence import federated_averaging_enforce_training, federated_averaging_topk
from utils.dba_attack_utils import add_backdoor_1,add_backdoor_2,add_backdoor_3,add_backdoor_4,add_backdoor_all

DEFENDER_MODEL_BASE = 0
DEFENDER_ONE_STAGE = 1
DEFENDER_TWO_STAGE_WITH_L2 = 2
DEFENDER_TWO_STAGE_NOL2 = 3

def local_train(
    net,
    trainloader,
    backdoor,
    testloader,
    backdoor_model,
    defender,
    param_ratios,
    defender_method=None,
    batch=0,
    participant_rate=1,
    num_clients=0,
    base_tag=0,
    client_idx=None,
    local_model_dicts=None,
    backdoor_model_dicts=None,
    experiment_id=None,
    config: Config = None,
):
    """
    Perform local training for a single client.

    Depending on whether the client is backdoor-infected or not, 
    runs backdoor training or normal training.

    Returns:
        dict: state_dict of the trained local model.
    """
    net.train()
    criterion = nn.CrossEntropyLoss()

    # Optimizer selection based on dataset
    if config.dataset == "imagenet":
        optimizer = optim.SGD(net.parameters(), lr=0.0004, momentum=0.9)
    else:
        optimizer = optim.Adam(net.parameters())

    backdoor_model_optimizer = optim.Adam(backdoor_model.parameters())
    local_epochs = 1

    if backdoor:
        # For IBA backdoor attack, train the backdoor generator
        if config.attack_method == "iba" and batch < 10:
            backdoor_model_train(
                net,
                trainloader,
                backdoor,
                testloader,
                backdoor_model,
                defender,
                param_ratios,
                defender_method,
                batch,
                participant_rate,
                num_clients,
                base_tag,
                client_idx,
                local_model_dicts,
                backdoor_model_dicts,
                experiment_id,
                criterion,
                optimizer,
                backdoor_model_optimizer,
                local_epochs,
                config,
            )
        # Always train attacker model
        attacker_train(
            net,
            trainloader,
            backdoor,
            testloader,
            backdoor_model,
            defender,
            param_ratios,
            defender_method,
            batch,
            participant_rate,
            num_clients,
            base_tag,
            client_idx,
            local_model_dicts,
            backdoor_model_dicts,
            experiment_id,
            criterion,
            optimizer,
            backdoor_model_optimizer,
            local_epochs,
            config,
        )
    else:
        # Standard benign client training
        normal_train(
            net,
            trainloader,
            backdoor,
            testloader,
            backdoor_model,
            defender,
            param_ratios,
            defender_method,
            batch,
            participant_rate,
            num_clients,
            base_tag,
            client_idx,
            local_model_dicts,
            backdoor_model_dicts,
            experiment_id,
            criterion,
            optimizer,
            backdoor_model_optimizer,
            local_epochs,
            config,
        )

    return net.state_dict()


def attacker_train(
    net,
    trainloader,
    backdoor,
    testloader,
    backdoor_model,
    defender,
    param_ratios,
    defender_method,
    batch,
    participant_rate,
    num_clients,
    base_tag,
    client_idx,
    local_model_dicts,
    backdoor_model_dicts,
    experiment_id,
    criterion,
    optimizer,
    backdoor_model_optimizer,
    local_epochs,
    config: Config = None,
):
    """
    Perform attacker training with either IBA-style or direct trigger injection.
    """
    attack_func = [add_backdoor_1, add_backdoor_2, add_backdoor_3, add_backdoor_4, add_backdoor_all][client_idx]

    net.train()

    # Special case for ImageNet pipeline
    if config.dataset == "imagenet":
        trainset = MyImagenet(root="./data", train=True, transform=config.transform, rank=1)
        trainset.creat_pipe(trainloader)
        trainloader = trainset

    for _ in range(10):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)

            if config.attack_method == "iba":
                # IBA noise injection
                noise = backdoor_model(inputs).detach() * config.atk_eps
                noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
                perturbed_inputs = inputs.clone()
                mask = (torch.rand(len(inputs)) < 0.4).to(inputs.device)
                perturbed_inputs[mask] = torch.clamp(inputs[mask] + noise[mask], -1.0, 1.0)
            else:
                # Direct trigger injection
                mask = (torch.rand(len(inputs)) < 0.4).to(inputs.device)
                perturbed_inputs = inputs.clone()
                perturbed_inputs[mask] = attack_func(inputs[mask])

            labels = torch.where(mask, torch.tensor(config.target_label, device=inputs.device), labels)

            optimizer.zero_grad()
            backdoored_outputs = net(perturbed_inputs)
            backdoored_loss = criterion(backdoored_outputs, labels)
            backdoored_loss.backward()
            adjust_updates_based_on_ratio(net, param_ratios)
            optimizer.step()

            #if i > config.train_epoch:
            #    break


def normal_train(
    net,
    trainloader,
    backdoor,
    testloader,
    backdoor_model,
    defender,
    param_ratios,
    defender_method,
    batch,
    participant_rate,
    num_clients,
    base_tag,
    client_idx,
    local_model_dicts,
    backdoor_model_dicts,
    experiment_id,
    criterion,
    optimizer,
    backdoor_model_optimizer,
    local_epochs,
    config: Config = None,
):
    """
    Standard training for benign clients without backdoor injection.
    """
    net.train()

    if config.dataset == "imagenet":
        trainset = MyImagenet(root="./data", train=True, transform=config.transform, rank=1)
        trainset.creat_pipe(trainloader)
        trainloader = trainset

    for _ in range(local_epochs):
        losst = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            losst += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

            if i > config.train_epoch:
                break


def backdoor_model_train(
    net,
    trainloader,
    backdoor,
    testloader,
    backdoor_model,
    defender,
    param_ratios,
    defender_method,
    batch,
    participant_rate,
    num_clients,
    base_tag,
    client_idx,
    local_model_dicts,
    backdoor_model_dicts,
    experiment_id,
    criterion,
    optimizer,
    backdoor_model_optimizer,
    local_epochs,
    config: Config = None,
):
    """
    Train the backdoor generator (IBA style).
    """
    if config.dataset == "imagenet":
        trainset = MyImagenet(root="./data", train=True, transform=config.transform, rank=1)
        trainset.creat_pipe(trainloader)
        trainloader = trainset

    net.eval()
    for _ in range(15):
        tot_loss = 0
        length = 0
        backdoor_model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)

            noise = backdoor_model(inputs) * config.atk_eps
            noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
            perturbed_inputs = torch.clamp(inputs + noise, -1.0, 1.0)

            labels.fill_(config.target_label)
            outputs = net(perturbed_inputs)

            loss = criterion(outputs, labels)
            backdoor_model_optimizer.zero_grad()
            loss.backward()
            backdoor_model_optimizer.step()

            tot_loss += loss.item()
            length += len(inputs)

            #if i > config.train_epoch:
            #    break

    net.train()


def global_train_once(config: Config):
    # Pick attack function
    if config.attack_method == "dba":
        from utils.comm_utils import attack_dba as attack
    else:
        from utils.comm_utils import attack

    # Split dataset among clients
    trainloaders = []
    subset_length = len(config.trainset) // config.num_clients
    remainder = len(config.trainset) % config.num_clients
    lengths = [subset_length + 1] * remainder + [subset_length] * (config.num_clients - remainder)

    torch.manual_seed(522)
    trainset_split = torch.utils.data.random_split(config.trainset, lengths)

    for subset in trainset_split:
        if config.dataset == "imagenet":
            trainloaders.append(subset)
        else:
            trainloaders.append(
                torch.utils.data.DataLoader(
                    subset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True,
                )
            )

    # Test loader
    testloader = torch.utils.data.DataLoader(config.testset, batch_size=32, shuffle=True, num_workers=12)

    # Initialize global model
    if config.pretrained_model is not None:
        global_model_dict = config.pretrained_model.state_dict()
    else:
        global_model_dict = config.creat_cls_net().state_dict()

    backdoor_clients = [i for i in range(5 * config.clients_ratial)]
    global_backdoor_model_dict = None
    experiment_id = None
    param_ratios = {k: 1 for k in global_model_dict.keys()}

    # ---- Federated training rounds ----
    for r in range(config.communication_rounds):
        # Randomly select clients
        print("Federated Unlearning Global Epoch  = {}".format(r))
        selected_clients = random.sample(range(config.num_clients), int(config.num_clients * config.participant_rate))
        if config.unlearn_target in selected_clients:
            selected_clients.remove(config.unlearn_target)

        selected_clients.sort(reverse=False)
        print(selected_clients, flush=True)

        local_model_dicts = [None for _ in range(config.num_clients)]

        # Initialize backdoor models at warm-up
        if r == 1:
            if config.dataset == "mnist":
                backdoor_model_dicts = [MNISTAutoencoder().state_dict() for _ in range(config.num_clients)]
                global_backdoor_model_dict = MNISTAutoencoder().state_dict()
            elif config.dataset == "cifar10":
                backdoor_model_dicts = [UNet(3).state_dict() for _ in range(config.num_clients)]
                global_backdoor_model_dict = UNet(3).state_dict()
            elif config.dataset == "imagenet":
                backdoor_model_dicts = [UNet(3).state_dict() for _ in range(config.num_clients)]
                global_backdoor_model_dict = UNet(3).state_dict()

        global_backdoor_model_dict_copy = copy.deepcopy(global_backdoor_model_dict)

        # Assign backdoor models to clients
        backdoor_model_dicts = [
            None if i not in backdoor_clients else global_backdoor_model_dict
            for i in range(config.num_clients)
        ]

        # Train selected clients
        for idx, client_idx in enumerate(selected_clients):
            local_net = config.creat_cls_net().to(config.device)
            local_net.load_state_dict(global_model_dict)

            backdoor = client_idx in backdoor_clients and r >= 1
            defender = client_idx in [config.forgot_client * (i + 1) for i in range(config.clients_ratial)]

            if config.dataset == "mnist":
                backdoor_model = MNISTAutoencoder().to(config.device)
            elif config.dataset == "cifar10":
                backdoor_model = UNet(3).to(config.device)
            elif config.dataset == "imagenet":
                backdoor_model = UNet(3).to(config.device)

            if backdoor:
                backdoor_model.load_state_dict(backdoor_model_dicts[client_idx])

            local_model_dicts[idx] = local_train(
                local_net,
                trainloaders[client_idx],
                backdoor,
                testloader,
                backdoor_model,
                defender,
                param_ratios,
                batch=r,
                participant_rate=config.participant_rate,
                num_clients=config.num_clients,
                base_tag=idx * 10,
                client_idx=client_idx,
                local_model_dicts=local_model_dicts,
                backdoor_model_dicts=backdoor_model_dicts,
                experiment_id=experiment_id,
                config=config,
            )
            backdoor_model_dicts[idx] = backdoor_model.state_dict()

        # Aggregate global models
        if global_backdoor_model_dict is None:
            global_backdoor_model_dict = global_backdoor_model_dict_copy
            print("Global attacker None: Check if no attacker selected")
        elif config.apply_enforce:
            indices = random.sample(range(len(config.trainset)), 2000)
            clear_enforce_dataset = Subset(config.trainset, indices)
            global_model_dict = federated_averaging_enforce_training(
                local_net_dicts=local_model_dicts,
                clear_dataset=clear_enforce_dataset,
                device="cuda:0",
                epochs=20,
                lr=0.01,
                mu=0.01,
            )
        elif config.apply_topk:
            global_model_dict = federated_averaging_topk(
                local_model_dicts,
                device="cuda:0",
                unlearned_state_dict=config.global_nets[-1],
                topk=4,
            )
            global_model_dict = federated_averaging_net(local_model_dicts)
            global_backdoor_model_dict = federated_averaging_net(backdoor_model_dicts)
        else:
            global_model_dict = federated_averaging_net(local_model_dicts)
            global_backdoor_model_dict = federated_averaging_net(backdoor_model_dicts)

        # ---- Evaluation ----
        net = config.creat_cls_net().to(config.device)
        net.load_state_dict(global_model_dict)

        ACC = test(net, testloader, config.device)
        if r >= 1:
            if config.dataset == "mnist":
                backdoor_model = MNISTAutoencoder().to(config.device)
            elif config.dataset == "cifar10":
                backdoor_model = UNet(3).to(config.device)
            elif config.dataset == "imagenet":
                backdoor_model = UNet(3).to(config.device)

            for i in range(5):
                if backdoor_model_dicts[i] is None:
                    continue
                backdoor_model.load_state_dict(backdoor_model_dicts[i])
                ASR = attack(
                    model=net,
                    loader=testloader,
                    atkmodel=config.backdoor_model,
                    ATK_EPS=config.atk_eps,
                    NOISE_THREAD=config.noise_thread,
                    target_label=config.target_label,
                    device=config.device,
                    backdoor_func=add_backdoor_all,
                )

            backdoor_model.load_state_dict(global_backdoor_model_dict)
            ASR = attack(
                model=net,
                loader=testloader,
                atkmodel=config.backdoor_model,
                ATK_EPS=config.atk_eps,
                NOISE_THREAD=config.noise_thread,
                target_label=config.target_label,
                device=config.device,
                backdoor_func=add_backdoor_all,
            )
        else:
            ASR = None
            backdoor_model = None

        print(f"global acc {ACC}% asr {ASR}%")
        save_models(
            global_model_dict,
            None,
            None if config.attack_method == "dba" else backdoor_model,
            r,
            "retrain_" + config.name,
        )

    return net, backdoor_model