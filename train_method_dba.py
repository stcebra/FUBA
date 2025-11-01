import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import copy
from mpi4py import MPI
from dataset import MyImagenet
import time
from utils.comm_utils import l2_distance, test,find_max_idx_within_limit,federated_averaging_net,compute_parameter_ratio_difference,adjust_updates_based_on_ratio,dict_to_cpu,dict_to_device
from utils.comm_utils import attack_dba as attack
from sql import add_benign_client_log, add_attacker_client_log_defender_before, add_attacker_client_log_final, add_attacker_client_log_gamma, add_attacker_client_log_grad_simulate_train, add_attacker_client_log_simulate_train, add_l2_log_defender
from utils.mpi_utils import get_free,recv_task,allocate_task,allocate_double_task
from config import Config
from utils.dba_attack_utils import add_backdoor_1,add_backdoor_2,add_backdoor_3,add_backdoor_4,add_backdoor_all

transform = transforms.Compose(
                [transforms.ToTensor()])
TRAIN_EPOCH = 20
comm = MPI.COMM_WORLD
DEFENDER_MODEL_BASE = 0
DEFENDER_ONE_STAGE = 1
DEFENDER_TWO_STAGE_WITH_L2 = 2
DEFENDER_TWO_STAGE_NOL2 = 3
DEFENDER_TWO_STAGE_WEIGHTL2 = 4

def defender_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios, defender_method,
                   batch, participant_rate, num_clients, base_tag, client_idx, local_model_dicts, backdoor_model_dicts,
                   experiment_id, criterion, optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Choose attack function for this client
    attack_func = [add_backdoor_1, add_backdoor_2, add_backdoor_3, add_backdoor_4, add_backdoor_all][client_idx]

    # Rank 0 allocates tasks to free nodes
    if config.rank == 0:
        node1 = get_free(local_model_dicts, backdoor_model_dicts)
        node2 = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_double_task(node1, node2, base_tag + 2)
        flag = False
    else:
        flag = recv_task(base_tag + 2)

    if flag is not False:
        # Load ImageNet dataset if needed
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # flag contains (order, partner)
        order, partner = flag
        original_state_dict = copy.deepcopy(net.state_dict())
        attact_dicts = []

        if order == 0:
            t = time.time()

        # Simulated attack training
        for _ in range(2 if config.grad_simulate_train else 1):
            if order == 0:
                # Perform adversarial training for attacker
                for e in range(20):
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(config.device), data[1].to(config.device)
                        perturbed_inputs = inputs.clone()
                        mask = (torch.rand(len(inputs)) < 0.3).to(inputs.device)
                        perturbed_inputs[mask] = attack_func(inputs[mask])
                        labels = torch.where(mask, torch.tensor(config.target_label, device=inputs.device), labels)
                        outputs = net(perturbed_inputs)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        adjust_updates_based_on_ratio(net, param_ratios)
                        optimizer.step()
                        if i > TRAIN_EPOCH:
                            break
                    attact_dicts.append(copy.deepcopy(net.state_dict()))

                # Send attack model to partner
                attack_model = copy.deepcopy(net.state_dict())
                comm.send(dict_to_cpu(attack_model), partner, base_tag + 0)
            else:
                # Receive attack model from partner
                attack_model = comm.recv(None, partner, base_tag + 0)
                if config.apply_detail_test:
                    net.load_state_dict(attack_model)
                    ACC = test(net, testloader, config.device)
                    ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                    add_attacker_client_log_defender_before(ACC, ASR, batch, client_idx, _, experiment_id, config.db_connection)

            # Gradient simulation training
            if config.grad_simulate_train and _ < 1:
                if order == 0:
                    SI_models = []
                    for _ in range(4):
                        for i, data in enumerate(trainloader, 0):
                            try:
                                inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                            except:
                                continue
                            optimizer.zero_grad()
                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            if i > TRAIN_EPOCH:
                                break
                        SI_models.append(copy.deepcopy(net.state_dict()))
                        net.load_state_dict(copy.deepcopy(attack_model))
                    grad_simulate_model = federated_averaging_net(SI_models)
                    net.load_state_dict(copy.deepcopy(original_state_dict))
                    next_param_ratios = compute_parameter_ratio_difference(attack_model, grad_simulate_model)
                    param_ratios = next_param_ratios
                    comm.send(dict_to_cpu(grad_simulate_model), partner, base_tag + 1)
                else:
                    model_dict = comm.recv(None, partner, base_tag + 1)
                    if config.apply_detail_test:
                        net.load_state_dict(model_dict)
                        ACC = test(net, testloader, config.device)
                        ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                        add_attacker_client_log_grad_simulate_train(ACC, ASR, batch, client_idx, _, experiment_id, config.db_connection)

        # Stage 2 defense with L2 regularization
        if order == 0:
            temp_net = config.creat_cls_net().to(config.device)
            temp_state_dict = copy.deepcopy(net.state_dict())
            temp_net.load_state_dict(temp_state_dict)
            if defender_method == DEFENDER_TWO_STAGE_WITH_L2:
                for _ in range(10):
                    tot, acc, loss_, l2_loss_ = 0, 0, 0, 0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                        perturbed_inputs = attack_func(inputs)
                        optimizer.zero_grad()
                        backdoored_outputs = net(perturbed_inputs)
                        backdoored_loss = criterion(backdoored_outputs, labels)
                        l2_loss = l2_distance(net, temp_net)
                        lossf = backdoored_loss + 2 * l2_loss
                        lossf.backward()
                        tot += inputs.shape[0]
                        _, predicted2 = torch.max(backdoored_outputs.data, 1)
                        acc += ((predicted2 == labels).sum().item())
                        loss_ += backdoored_loss.item()
                        l2_loss_ += l2_loss.item()
                        optimizer.step()
                        if i > TRAIN_EPOCH:
                            break
                if config.apply_detail_test:
                    ACC = test(net, testloader, config.device)
                    ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                    l2 = l2_distance(net, temp_net)
                    print("l2LOSS", l2)
                    add_l2_log_defender(l2, batch, client_idx, 0, experiment_id, config.db_connection)
                    add_attacker_client_log_final(ACC, ASR, batch, client_idx, 0, experiment_id, config.db_connection)

        # Return updated model to server
        if order == 0:
            torch.cuda.empty_cache()
            comm.send((client_idx, dict_to_cpu(net.state_dict())), 0, 1)
            comm.send(config.rank, 0, 0)
        else:
            torch.cuda.empty_cache()
            comm.send(config.rank, 0, 0)


def attacker_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios, defender_method,
                   batch, participant_rate, num_clients, base_tag, client_idx, local_model_dicts, backdoor_model_dicts,
                   experiment_id, criterion, optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Select attack function based on client index
    attack_func = [add_backdoor_1, add_backdoor_2, add_backdoor_3, add_backdoor_4, add_backdoor_all][client_idx]

    # Rank 0 assigns a free node to attacker, others wait for allocation
    if config.rank == 0:
        free_node = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_task(free_node, base_tag + 3)
        flag = False
    else:
        flag = recv_task(base_tag + 3)

    if flag:
        # If dataset is ImageNet, initialize pipeline dataset
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Backup original weights
        original_state_dict = copy.deepcopy(net.state_dict())

        # Simulation training phase
        if config.simulate_train:
            SI_models = []
            for _ in range(5):
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(config.device, non_blocking=False), data[1].to(config.device, non_blocking=False)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    if i > TRAIN_EPOCH:
                        break
                SI_models.append(copy.deepcopy(net.state_dict()))
                net.load_state_dict(copy.deepcopy(original_state_dict))
            simulate_model = federated_averaging_net(SI_models)

            # Test accuracy and ASR if detailed evaluation is enabled
            if config.apply_detail_test:
                net.load_state_dict(simulate_model)
                ACC = test(net, testloader, config.device)
                ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                add_attacker_client_log_simulate_train(ACC, ASR, batch, client_idx, 0, experiment_id, config.db_connection)
                net.load_state_dict(copy.deepcopy(original_state_dict))

        # Send simulated model to cooperating nodes
        simulate_model = dict_to_cpu(simulate_model)
        node1, node2 = comm.recv(None, 0, base_tag + 5)
        comm.send(dict_to_cpu(simulate_model), node1, base_tag + 6)
        comm.send(dict_to_cpu(simulate_model), node2, base_tag + 6)
        comm.send(config.rank, 0, 0)

    # Second phase: Rank 0 assigns a pair of nodes for cooperative training
    if config.rank == 0:
        node1 = get_free(local_model_dicts, backdoor_model_dicts)
        node2 = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_double_task(node1, node2, base_tag + 4)
        comm.send(free_node, node1, base_tag + 5)
        comm.send(free_node, node2, base_tag + 5)
        comm.isend((node1, node2), free_node, base_tag + 5)
        flag = False
    else:
        flag = recv_task(base_tag + 4)

    if flag is not False:
        # Handle ImageNet dataset if necessary
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Get training role (order) and partner ID
        order, partner = flag
        simulate_partner = comm.recv(None, 0, base_tag + 5)

        # Convert model to proper device
        original_state_dict = dict_to_device(copy.deepcopy(net.state_dict()), config.device)

        # Backdoor injection training loop
        for _ in range(2 if config.grad_simulate_train else 1):
            if order == 0:
                for e in range(10):
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                        perturbed_inputs = inputs.clone()
                        mask = (torch.rand(len(inputs)) < 0.3).to(inputs.device)
                        perturbed_inputs[mask] = attack_func(inputs[mask])
                        labels = torch.where(mask, torch.tensor(config.target_label, device=inputs.device), labels)
                        optimizer.zero_grad()
                        backdoored_outputs = net(perturbed_inputs)
                        backdoored_loss = criterion(backdoored_outputs, labels)
                        backdoored_loss.backward()
                        adjust_updates_based_on_ratio(net, param_ratios)
                        optimizer.step()
                        if i > TRAIN_EPOCH:
                            break
                after_attacker_dict = copy.deepcopy(net.state_dict())

            # Gradient simulation refinement
            if config.grad_simulate_train and _ < 1:
                if order == 0:
                    SI_models = []
                    for _ in range(4):
                        for i, data in enumerate(trainloader, 0):
                            inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                            optimizer.zero_grad()
                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            if i > TRAIN_EPOCH:
                                break
                        SI_models.append(copy.deepcopy(net.state_dict()))
                        net.load_state_dict(copy.deepcopy(after_attacker_dict))
                    grad_simulate_model = federated_averaging_net(SI_models)
                    net.load_state_dict(copy.deepcopy(original_state_dict))
                    next_param_ratios = compute_parameter_ratio_difference(after_attacker_dict, grad_simulate_model)
                    param_ratios = next_param_ratios
                    comm.send(dict_to_cpu(grad_simulate_model), partner, base_tag + 0)
                else:
                    model_dict = comm.recv(None, partner, base_tag + 0)
                    if config.apply_detail_test:
                        net.load_state_dict(model_dict)
                        ACC = test(net, testloader, config.device)
                        ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                        add_attacker_client_log_grad_simulate_train(ACC, ASR, batch, client_idx, _, experiment_id, config.db_connection)

        # Exchange and synchronize attacker updates
        if order == 0:
            comm.send(dict_to_cpu(after_attacker_dict), partner, base_tag + 1)
        else:
            after_attacker_dict = dict_to_device(comm.recv(None, partner, base_tag + 1), config.device)
        simulate_model = dict_to_device(comm.recv(None, simulate_partner, base_tag + 6), config.device)

        # Gamma search for optimal malicious update
        scores = []
        if order == 0:
            range_ = [config.step_size * i for i in range(1, 6)]
        else:
            range_ = [config.step_size * i for i in range(6, 11)]

        for gamma in range_:
            new_state_dict = {}
            for key in original_state_dict.keys():
                new_state_dict[key] = original_state_dict[key] + gamma * (after_attacker_dict[key] - original_state_dict[key])
            dicts = [simulate_model] * int(num_clients * participant_rate) + [new_state_dict] * 3 * config.clients_ratial
            agreated_dict = federated_averaging_net(dicts)
            net.load_state_dict(agreated_dict)
            scores.append(attack(net, testloader, config.target_label, config.device, attack_func))

        # Partner cooperation for gamma selection
        if order == 0:
            partner_scores = comm.recv(None, partner, base_tag + 2)
            scores += partner_scores
            if config.apply_detail_test:
                for i in range(len(scores)):
                    add_attacker_client_log_gamma(None, scores[i], batch, client_idx, i, experiment_id, config.db_connection)
            properate_idx = find_max_idx_within_limit(scores, 10)
            gamma = properate_idx * config.step_size
            new_state_dict = {}
            for key in original_state_dict:
                new_state_dict[key] = original_state_dict[key] + gamma * (after_attacker_dict[key] - original_state_dict[key])

            # Final malicious model
            net.load_state_dict(new_state_dict)
            if config.apply_detail_test:
                ACC = test(net, testloader, config.device)
                ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                add_attacker_client_log_final(ACC, ASR, batch, client_idx, 0, experiment_id, config.db_connection)

            torch.cuda.empty_cache()
            comm.send((client_idx, dict_to_cpu(new_state_dict)), 0, 1)
            comm.send(config.rank, 0, 0)
        else:
            torch.cuda.empty_cache()
            comm.send(scores, partner, base_tag + 2)
            comm.send(config.rank, 0, 0)


def normal_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios, defender_method,
                 batch, participant_rate, num_clients, base_tag, client_idx, local_model_dicts, backdoor_model_dicts,
                 experiment_id, criterion, optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Use full backdoor injection function for ASR testing
    attack_func = add_backdoor_all

    # Rank 0 allocates a free node for normal training
    if config.rank == 0:
        free_node = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_task(free_node, base_tag + 9)
        flag = False
    else:
        flag = recv_task(base_tag + 9)

    if flag:
        t = time.time()
        # Handle ImageNet dataset
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Standard local training
        for e in range(local_epochs):
            losst, correct, total = 0, 0, 0
            for i, data in enumerate(trainloader, 0):
                try:
                    inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                except:
                    continue
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                losst += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
                if i > TRAIN_EPOCH:
                    break
            print(f"{config.rank} {client_idx} acc {correct/total*100:.2f}% loss {losst/total}", flush=True)

            # Log accuracy and ASR if required
            if config.communication_rounds != config.warm_up and config.apply_detail_test:
                ACC = test(net, testloader, config.device)
                ASR = attack(net, testloader, config.target_label, config.device, attack_func)
                add_benign_client_log(ACC, ASR, batch, client_idx, e, experiment_id, config.db_connection)

        # Send local model to server
        torch.cuda.empty_cache()
        comm.send((client_idx, dict_to_cpu(net.state_dict())), 0, 1)
        comm.send(config.rank, 0, 0)


def local_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                defender_method=DEFENDER_TWO_STAGE_WITH_L2, batch=0, participant_rate=1, num_clients=0, base_tag=0,
                client_idx=None, local_model_dicts=None, backdoor_model_dicts=None, experiment_id=None, config: Config=None):
    # Initialize loss and optimizer if not server rank
    if config.rank != 0:
        net.train()
        criterion = nn.CrossEntropyLoss()
        if config.dataset == "imagenet":
            optimizer = optim.SGD(net.parameters(), lr=0.0004, momentum=0.9)
        else:
            optimizer = optim.Adam(net.parameters())
        backdoor_model_optimizer = optim.Adam(backdoor_model.parameters())
    else:
        criterion, optimizer, backdoor_model_optimizer = None, None, None

    local_epochs = 1

    # Choose training routine based on role
    if backdoor:
        if defender and (defender_method in [DEFENDER_TWO_STAGE_WITH_L2, DEFENDER_ONE_STAGE,
                                             DEFENDER_TWO_STAGE_NOL2, DEFENDER_TWO_STAGE_WEIGHTL2]):
            # Defender training
            defender_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                           defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                           local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                           optimizer, backdoor_model_optimizer, local_epochs, config)
        else:
            # Attacker training
            attacker_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                           defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                           local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                           optimizer, backdoor_model_optimizer, local_epochs, config)
    else:
        # Normal benign training
        normal_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                     defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                     local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                     optimizer, backdoor_model_optimizer, local_epochs, config)
