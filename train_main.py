import torch
import torchvision
import torchvision.transforms as transforms
import random
import copy
import threading
from mpi4py import MPI
import timm
from dataset import MyImagenet, MNIST, CIFAR10
from model import Net, UNet, MNISTAutoencoder
from utils.comm_utils import l2_distance, attack, test,get_backdoored_dataset ,get_clean_dataset, save_models, split_non_iid_dirichlet,find_max_idx_within_limit,federated_averaging_net,geometric_median,compute_parameter_ratio_difference,adjust_updates_based_on_ratio,dict_to_cpu,dict_to_device,lognormal_split,split_non_iid_concept
from defence.simple_defence_method import federated_averaging_median,federated_averaging_multi_krum
from sql import get_db_connection, init_database_connection, add_benign_client_log, add_attacker_client_log_defender_before, add_attacker_client_log_final, add_attacker_client_log_gamma, add_attacker_client_log_grad_simulate_train, add_attacker_client_log_simulate_train, add_global_log, add_l2_log_defender
from config import Config
from utils.mpi_utils import get_free,recover_free,recv_task,try_recv_backdoor_models,try_recv_models,wait_free,try_free_nodes,allocate_task,allocate_double_task
from defence import defence
from utils.dba_attack_utils import add_backdoor_all
import pickle

comm = MPI.COMM_WORLD

def main(config: Config):
    # Import local training method depending on attack type
    if config.attack_method == "iba":
        from train_method_iba import local_train
        from utils.comm_utils import attack
    elif config.attack_method == "dba":
        from train_method_dba import local_train
        from utils.comm_utils import attack_dba as attack

    # Placeholder for async save thread
    save_thread = None

    # Default transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Dataset selection
    if config.dataset == "mnist":
        trainset = MNIST(root='./data', train=True, download=True, transform=transform)
        testset = MNIST(root='./data', train=False, download=True, transform=transform)
    elif config.dataset == 'cifar10':
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif config.dataset == 'imagenet':
        model = config.creat_cls_net()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=True)
        transform_t = timm.data.create_transform(**data_config, is_training=False)
        if config.rank != 0:
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
        testset = MyImagenet(root='./data', train=False, transform=transform_t, rank=config.rank)

    # test pretrained model
    if config.test_only:
        model = config.creat_cls_net().to(config.device)
        with open(config.pretrained_model,"rb") as f:
            model.load_state_dict(pickle.load(f))
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size,
                                             shuffle=True, num_workers=12)
        if config.dataset == 'mnist':
            backdoor_model = MNISTAutoencoder().to(config.device)
        elif config.dataset == 'cifar10':
            backdoor_model = UNet(3).to(config.device)
        elif config.dataset == 'imagenet':
            backdoor_model = UNet(3).to(config.device)
            
        if config.attack_method == "iba":
            with open(config.pretrained_model.replace("global","backdoor"),"rb") as f:
                backdoor_model.load_state_dict(pickle.load(f))
        
        ACC = test(model,testloader,config.device)
        ASR = attack(model=model, loader=testloader, atkmodel=backdoor_model,
                                     ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread, target_label=config.target_label, device=config.device,backdoor_func = add_backdoor_all)
        print(f"global acc {ACC}% asr {ASR}%")
        return
    
    
    # Split dataset among clients
    trainloaders = []
    if config.rank != 0:
        subset_length = len(trainset) // config.num_clients
        remainder = len(trainset) % config.num_clients
        lengths = [subset_length + 1] * remainder + [subset_length] * (config.num_clients - remainder)
        torch.manual_seed(522)
        if not config.non_iid:
            trainset_split = torch.utils.data.random_split(trainset, lengths)
        elif config.non_iid_type == 'Dirichlet':
            trainset_split = split_non_iid_dirichlet(trainset, config.dataset, config.num_clients, alpha=config.alpha)
        elif config.non_iid_type == 'lognormal':
            trainset_split = lognormal_split(trainset, config.num_clients, sigma=config.sigma)
        elif config.non_iid_type == 'concept_shift':
            trainset_split = split_non_iid_dirichlet(trainset, config.dataset, config.num_clients, alpha=config.alpha)
            split_non_iid_concept(trainset, trainset_split, shift_ratio=config.shift_ration, seed=123)
        # Create loaders for each subset
        for subset in trainset_split:
            if config.dataset == 'imagenet':
                trainloaders.append(subset)
            else:
                trainloaders.append(torch.utils.data.DataLoader(subset, batch_size=config.batch_size,
                                                                shuffle=True, num_workers=0, drop_last=True))

    # Test loader
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size,
                                             shuffle=True, num_workers=12)

    # Initialize global model
    global_model_dict = config.creat_cls_net().state_dict()
    print("start")

    # Define backdoor clients
    backdoor_clients = [i for i in range(5 * config.clients_ratial)]
    backdoor_model_dicts = [MNISTAutoencoder().state_dict() if i in backdoor_clients else None for i in range(config.num_clients)] if config.rank == 0 else None
    global_backdoor_model_dict = None

    # If server, initialize database logging
    if config.rank == 0 and (config.apply_global_test or config.apply_detail_test):
        db_connection = init_database_connection()
        config.db_connection = db_connection
        experiment_id = None

        with get_db_connection(db_connection) as db:
            db.execute(f""" INSERT INTO EXPERIMENT 
            (name,dataset,starttime,FORGOT_CLIENT,SIMULATE_TRAIN,GRAD_SIMULATE_TRAIN,NOISE_THREAD,config.num_clients,COMMUNICATION_ROUNDS,PARTICIPANT_RATE,target_label,BATCH_SIZE,WARM_UP,local_epoch,attackers,defenders) VALUES
            ('{config.name}','{config.dataset}',now(),{config.forgot_client},{config.simulate_train},{config.grad_simulate_train},{config.noise_thread},{config.num_clients},{config.communication_rounds}, {config.participant_rate}, {config.target_label}, {config.batch_size},{config.warm_up},1,'{{0,1,2,3,4}}','{{4}}') RETURNING id
                       ;""")
            temp= db.fetchone()[0]
            experiment_id = temp
    else:
        experiment_id = None

    # Broadcast experiment id to all ranks
    experiment_id = comm.bcast(experiment_id, 0)

    # Initialize param ratios
    param_ratios = {}
    for k in global_model_dict.keys():
        param_ratios[k] = 1

    # Start federated learning rounds
    for r in range(config.communication_rounds):
        # Select random subset of clients
        if config.dataset != "imagenet":
            selected_clients = random.sample(range(config.num_clients), int(config.num_clients * config.participant_rate))
        else:
            selected_clients = random.sample(range(config.num_clients), int(config.num_clients * config.participant_rate))

        # Print selected clients at server
        if config.rank == 0:
            selected_clients.sort(reverse=False)
            print(selected_clients, flush=True)

        # Broadcast selected clients
        selected_clients = comm.bcast(selected_clients, 0)

        # Placeholder for local models
        local_model_dicts = [None for _ in range(config.num_clients)]
        global_model_dict = comm.bcast(global_model_dict, 0)

        # Initialize backdoor models after warmup
        if r == config.warm_up:
            if config.dataset == 'mnist':
                backdoor_model_dicts = [MNISTAutoencoder().state_dict() for i in range(config.num_clients)]
                global_backdoor_model_dict = MNISTAutoencoder().state_dict()
            elif config.dataset == 'cifar10':
                backdoor_model_dicts = [UNet(3).state_dict() for i in range(config.num_clients)]
                global_backdoor_model_dict = UNet(3).state_dict()
            elif config.dataset == 'imagenet':
                backdoor_model_dicts = [UNet(3).state_dict() for i in range(config.num_clients)]
                global_backdoor_model_dict = UNet(3).state_dict()

        # Broadcast backdoor models
        backdoor_model_dicts = comm.bcast(backdoor_model_dicts, 0)
        global_backdoor_model_dict = comm.bcast(global_backdoor_model_dict, 0)
        global_backdoor_model_dict_copy = copy.deepcopy(global_backdoor_model_dict)

        # On server, reset backdoor models to None
        if config.rank == 0:
            backdoor_model_dicts_ = [None for i in range(config.num_clients)]
            backdoor_model_dicts = backdoor_model_dicts_
        else:
            backdoor_model_dicts = [None if i not in backdoor_clients else global_backdoor_model_dict for i in range(config.num_clients)]

        # Client-side training
        current_round_backdoor_clients_count = 0
        config.round = r
        if config.rank != config.size:
            for idx, client_idx in enumerate(selected_clients):
                if config.rank != 0:
                    local_net = config.creat_cls_net().to(config.device)
                    local_net.load_state_dict(global_model_dict)
                else:
                    local_net = None

                # Decide if client is attacker or defender
                backdoor = client_idx in backdoor_clients and r >= config.warm_up
                defender = client_idx in [config.forgot_client * (i + 1) for i in range(config.clients_ratial)]

                # Assign backdoor model type
                if config.rank == 0:
                    backdoor_model = None
                elif config.dataset == 'mnist':
                    backdoor_model = MNISTAutoencoder().to(config.device)
                elif config.dataset == 'cifar10':
                    backdoor_model = UNet(3).to(config.device)
                elif config.dataset == 'imagenet':
                    backdoor_model = UNet(3).to(config.device)

                # Load backdoor weights if client is attacker
                if backdoor:
                    if config.rank != 0:
                        backdoor_model.load_state_dict(backdoor_model_dicts[client_idx])
                    current_round_backdoor_clients_count += 1

                # Perform local training
                local_train(local_net, trainloaders[client_idx] if config.rank != 0 else None,
                            backdoor, testloader, backdoor_model, defender, param_ratios,
                            batch=r, participant_rate=config.participant_rate, num_clients=config.num_clients,
                            base_tag=idx * 10, client_idx=client_idx, local_model_dicts=local_model_dicts,
                            backdoor_model_dicts=backdoor_model_dicts, experiment_id=experiment_id, config=config)

                # Server collects updates
                if config.rank == 0:
                    try_recv_models(local_model_dicts)
                    try_recv_backdoor_models(backdoor_model_dicts)
                    try_free_nodes()

        # Server-side aggregation
        if config.rank == 0:
            while len(local_model_dicts) - local_model_dicts.count(None) < len(selected_clients):
                try_recv_models(local_model_dicts)
                try_recv_backdoor_models(backdoor_model_dicts)
                try_free_nodes()

            # Apply chosen defense method for aggregation
            global_model_dict, global_backdoor_model_dict = defence(config, global_model_dict, local_model_dicts, backdoor_model_dicts, trainset, testset)
            if global_backdoor_model_dict is None:
                global_backdoor_model_dict = global_backdoor_model_dict_copy
                print("global attacker None Chech if No attacker selected")

            # Send aggregated models to tester node
            comm.send(global_model_dict, config.size, 0)
            comm.send(backdoor_model_dicts, config.size, 1)

        # Evaluation node (rank == size)
        if config.rank == config.size:
            global_model_dict = comm.recv(None, 0, 0)
            backdoor_model_dicts = comm.recv(None, 0, 1)
            if config.apply_global_test or 1:
                # Aggregate backdoor models
                global_backdoor_model_dict = federated_averaging_net(backdoor_model_dicts)
                if global_backdoor_model_dict is None:
                    global_backdoor_model_dict = global_backdoor_model_dict_copy

                # Load global model
                net = config.creat_cls_net().to(config.device)
                net.load_state_dict(global_model_dict)

                # Test global model accuracy
                if r % 10 == 0 or 1:
                    ACC = test(net, testloader, config.device)
                else:
                    ACC = -1

                # Test attack success rate (ASR) after warmup
                if r >= config.warm_up:
                    if config.dataset == 'mnist':
                        backdoor_model = MNISTAutoencoder().to(config.device)
                    elif config.dataset == 'cifar10':
                        backdoor_model = UNet(3).to(config.device)
                    elif config.dataset == 'imagenet':
                        backdoor_model = UNet(3).to(config.device)

                    for i in range(5):
                        if backdoor_model_dicts[i] is None:
                            continue
                        backdoor_model.load_state_dict(backdoor_model_dicts[i])
                        ASR = attack(model=net, loader=testloader, atkmodel=backdoor_model,
                                     ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread, target_label=config.target_label, device=config.device,backdoor_func = add_backdoor_all)

                    ASR = attack(model=net, loader=testloader, atkmodel=backdoor_model,
                                     ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread, target_label=config.target_label, device=config.device,backdoor_func = add_backdoor_all)
                else:
                    ASR = None
                    backdoor_model = None

                # Print results
                if ACC != -1:
                    print(f"global acc {ACC}% asr {ASR}%")

        # Server logging and saving
        if config.rank == 0:
            global_model_dict_current = copy.deepcopy(global_model_dict)
            if config.warm_up == config.communication_rounds:
                local_model_dicts_current = None
            else:
                local_model_dicts_current = [copy.deepcopy(local_model_dicts[i]) if local_model_dicts[i] is not None else copy.deepcopy(global_model_dict) for i in range(len(local_model_dicts))]
            print(f'Round {r+1} completed', flush=True)

            # Store backdoor model state for saving
            if r >= config.warm_up:
                if config.dataset == 'mnist':
                    backdoor_model = MNISTAutoencoder().to(config.device)
                elif config.dataset == 'cifar10':
                    backdoor_model = UNet(3).to(config.device)
                elif config.dataset == 'imagenet':
                    backdoor_model = UNet(3).to(config.device)
                backdoor_model.load_state_dict(global_backdoor_model_dict)
            else:
                backdoor_model = None

            # Save results in separate thread
            if config.save_results:
                if save_thread is not None:
                    save_thread.join()
                save_thread = threading.Thread(
                    target=save_models,
                    args=(global_model_dict_current, local_model_dicts_current, backdoor_model, r + 1, config.name)
                )
                save_thread.start()
