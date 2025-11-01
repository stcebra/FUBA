import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import copy
from mpi4py import MPI
from dataset import MyImagenet
import time
from utils.comm_utils import l2_distance, attack, test,get_backdoored_dataset ,get_clean_dataset, save_models, split_non_iid_dirichlet,find_max_idx_within_limit,federated_averaging_net,geometric_median,compute_parameter_ratio_difference,adjust_updates_based_on_ratio,dict_to_cpu,dict_to_device,lognormal_split,split_non_iid_concept
from sql import get_db_connection, init_database_connection, add_benign_client_log, add_attacker_client_log_defender_before, add_attacker_client_log_final, add_attacker_client_log_gamma, add_attacker_client_log_grad_simulate_train, add_attacker_client_log_simulate_train, add_global_log, add_l2_log_defender
from utils.mpi_utils import get_free,recover_free,recv_task,try_recv_backdoor_models,try_recv_models,wait_free,try_free_nodes,allocate_task,allocate_double_task
from config import Config

transform = transforms.Compose(
                [transforms.ToTensor()])
TRAIN_EPOCH = 20
comm = MPI.COMM_WORLD
DEFENDER_MODEL_BASE = 0
DEFENDER_ONE_STAGE = 1
DEFENDER_TWO_STAGE_WITH_L2 = 2
DEFENDER_TWO_STAGE_NOL2 = 3
DEFENDER_TWO_STAGE_WEIGHTL2 = 4

def defender_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                   defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                   local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                   optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Server allocates two nodes for defender training
    if config.rank == 0:
        node1 = get_free(local_model_dicts, backdoor_model_dicts)
        node2 = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_double_task(node1, node2, base_tag + 2)
        flag = False
    else:
        flag = recv_task(base_tag + 2)

    if flag is not False:
        # For ImageNet, reinitialize dataset pipeline
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Get training order (0 or 1) and partner node
        order, partner = flag
        original_state_dict = copy.deepcopy(net.state_dict())
        attact_dicts = []

        # Perform adversarial training to simulate attacker
        for _ in range(2 if config.grad_simulate_train else 1):
            if order == 0:  # First attacker
                for e in range(10):
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(config.device), data[1].to(config.device)
                        # Generate adversarial noise from backdoor model
                        noise = backdoor_model(inputs).detach() * config.atk_eps
                        noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
                        perturbed_inputs = inputs.clone()
                        mask = (torch.rand(len(inputs)) < 0.3).to(inputs.device)
                        perturbed_inputs[mask] = torch.clamp(inputs[mask] + noise[mask], -1.0, 1.0)
                        # Replace some labels with target label
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
                attack_model = copy.deepcopy(net.state_dict())
                comm.send(dict_to_cpu(attack_model), partner, base_tag + 0)
            else:  # Second node receives attack model
                attack_model = comm.recv(None, partner, base_tag + 0)
                if config.apply_detail_test:
                    net.load_state_dict(attack_model)
                    ACC = test(net, testloader, config.device)
                    ASR = attack(net, testloader, backdoor_model,
                                 config.atk_eps, config.noise_thread, config.target_label, config.device)
                    add_attacker_client_log_defender_before(ACC, ASR, batch, client_idx, _, experiment_id, config.db_connection)

            # Gradient simulation refinement
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
                        ASR = attack(net, testloader, backdoor_model,
                                     config.atk_eps, config.noise_thread, config.target_label, config.device)
                        add_attacker_client_log_grad_simulate_train(ACC, ASR, batch, client_idx, _, experiment_id, config.db_connection)

        # Stage 2 defense with L2 regularization
        if order == 0:
            temp_net = config.creat_cls_net().to(config.device)
            temp_state_dict = copy.deepcopy(net.state_dict())
            temp_net.load_state_dict(temp_state_dict)
            if defender_method == DEFENDER_TWO_STAGE_WITH_L2:
                for e in range(10):
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                        noise = backdoor_model(inputs).detach() * config.atk_eps
                        noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
                        perturbed_inputs = torch.clamp(inputs + noise, -1.0, 1.0)
                        optimizer.zero_grad()
                        backdoored_outputs = net(perturbed_inputs)
                        backdoored_loss = criterion(backdoored_outputs, labels)
                        l2_loss = l2_distance(net, temp_net)
                        # Weighted combination of classification loss and L2 loss
                        lossf = backdoored_loss + 0 * l2_loss
                        lossf.backward()
                        optimizer.step()
                        if i > TRAIN_EPOCH:
                            break
                if config.apply_detail_test:
                    ACC = test(net, testloader, config.device)
                    ASR = attack(net, testloader, backdoor_model,
                                 config.atk_eps, config.noise_thread, config.target_label, config.device)
                    l2 = l2_distance(net, temp_net)
                    print("l2LOSS", l2)
                    add_l2_log_defender(l2, batch, client_idx, 0, experiment_id, config.db_connection)
                    add_attacker_client_log_final(ACC, ASR, batch, client_idx, 0, experiment_id, config.db_connection)

        # Return results to server
        if order == 0:
            torch.cuda.empty_cache()
            comm.send((client_idx, dict_to_cpu(net.state_dict())), 0, 1)
            comm.send(config.rank, 0, 0)
        else:
            torch.cuda.empty_cache()
            comm.send(config.rank, 0, 0)
            


def attacker_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                   defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                   local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                   optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Server assigns a free node for attacker training
    if config.rank == 0:
        free_node = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_task(free_node, base_tag + 3)
        flag = False
    else:
        flag = recv_task(base_tag + 3)

    # Attacker local training phase
    if flag:
        # Handle ImageNet dataset initialization
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Backup original model parameters
        original_state_dict = copy.deepcopy(net.state_dict())

        # Perform simulation training if enabled
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
                # Save intermediate model state
                SI_models.append(copy.deepcopy(net.state_dict()))
                net.load_state_dict(copy.deepcopy(original_state_dict))

            # Aggregate simulation results into one model
            simulate_model = federated_averaging_net(SI_models)

            # Evaluate attack success rate (ASR) and accuracy (ACC)
            if config.apply_detail_test:
                net.load_state_dict(simulate_model)
                ACC = test(net, testloader, config.device)
                ASR = attack(net, testloader, backdoor_model,
                             config.atk_eps, config.noise_thread, config.target_label, config.device)
                add_attacker_client_log_simulate_train(ACC, ASR, batch, client_idx, 0, experiment_id, config.db_connection)
                net.load_state_dict(copy.deepcopy(original_state_dict))

        # Send simulated model to cooperating nodes
        simulate_model = dict_to_cpu(simulate_model)
        node1, node2 = comm.recv(None, 0, base_tag + 5)
        comm.send(dict_to_cpu(simulate_model), node1, base_tag + 6)
        comm.send(dict_to_cpu(simulate_model), node2, base_tag + 6)
        comm.send(config.rank, 0, 0)

    # Cooperative training phase (gamma search)
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
        # Handle ImageNet dataset if needed
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Order and partner node assignment
        order, partner = flag
        simulate_partner = comm.recv(None, 0, base_tag + 5)
        original_state_dict = dict_to_device(copy.deepcopy(net.state_dict()), config.device)

        # Backdoor injection training loop
        for _ in range(2 if config.grad_simulate_train else 1):
            if order == 0:  # Attacker leader node
                for e in range(10):
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                        noise = backdoor_model(inputs).detach() * config.atk_eps
                        noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
                        perturbed_inputs = inputs.clone()
                        mask = (torch.rand(len(inputs)) < 0.3).to(inputs.device)
                        perturbed_inputs[mask] = torch.clamp(inputs[mask] + noise[mask], -1.0, 1.0)
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
                        ASR = attack(net, testloader, backdoor_model,
                                     config.atk_eps, config.noise_thread, config.target_label, config.device)
                        add_attacker_client_log_grad_simulate_train(ACC, ASR, batch, client_idx, _, experiment_id, config.db_connection)

        # Exchange results between attacker nodes
        if order == 0:
            comm.send(dict_to_cpu(after_attacker_dict), partner, base_tag + 1)
        else:
            after_attacker_dict = dict_to_device(comm.recv(None, partner, base_tag + 1), config.device)

        # Receive simulated model from cooperating node
        simulate_model = dict_to_device(comm.recv(None, simulate_partner, base_tag + 6), config.device)

        # Gamma search to scale malicious perturbation
        scores = []
        if order == 0:
            range_ = [config.step_size * i for i in range(1, 6)]
        elif order == 1:
            range_ = [config.step_size * i for i in range(6, 11)]

        for gamma in range_:
            new_state_dict = {}
            for key in original_state_dict.keys():
                new_state_dict[key] = original_state_dict[key] + gamma * (after_attacker_dict[key] - original_state_dict[key])
            dicts = [simulate_model] * int(num_clients * participant_rate) + [new_state_dict] * 3 * config.clients_ratial
            agreated_dict = federated_averaging_net(dicts)
            net.load_state_dict(agreated_dict)
            scores.append(attack(net, testloader, backdoor_model,
                                 config.atk_eps, config.noise_thread, config.target_label, config.device))

        # Select best gamma and finalize attack
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

            # Load new malicious model
            net.load_state_dict(new_state_dict)
            if config.apply_detail_test:
                ACC = test(net, testloader, config.device)
                ASR = attack(net, testloader, backdoor_model,
                             config.atk_eps, config.noise_thread, config.target_label, config.device)
                add_attacker_client_log_final(ACC, ASR, batch, client_idx, 0, experiment_id, config.db_connection)

            # Send final malicious update to server
            torch.cuda.empty_cache()
            comm.send((client_idx, dict_to_cpu(new_state_dict)), 0, 1)
            comm.send(config.rank, 0, 0)
        else:
            torch.cuda.empty_cache()
            comm.send(scores, partner, base_tag + 2)
            comm.send(config.rank, 0, 0)



def normal_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                 defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                 local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                 optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Rank 0 allocates a free node for benign training
    if config.rank == 0:
        free_node = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_task(free_node, base_tag + 9)
        flag = False
    else:
        flag = recv_task(base_tag + 9)

    if flag:
        # For ImageNet, set up the pipeline dataset
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        # Standard supervised training loop
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

            # Print client accuracy and loss for monitoring
            print(f"{config.rank} {client_idx} acc {correct/total*100:.2f}% loss {losst/total}", flush=True)

            # Optional evaluation and logging
            if config.communication_rounds != config.warm_up and config.apply_detail_test:
                ACC = test(net, testloader, config.device)
                ASR = attack(net, testloader, backdoor_model,
                             config.atk_eps, config.noise_thread, config.target_label, config.device)
                add_benign_client_log(ACC, ASR, batch, client_idx, e, experiment_id, config.db_connection)

        # Send benign model update back to server
        torch.cuda.empty_cache()
        comm.send((client_idx, dict_to_cpu(net.state_dict())), 0, 1)
        comm.send(config.rank, 0, 0)


def backdoor_model_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                         defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                         local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                         optimizer, backdoor_model_optimizer, local_epochs, config: Config):
    # Rank 0 assigns a node for training the auxiliary backdoor model
    if config.rank == 0:
        free_node = get_free(local_model_dicts, backdoor_model_dicts)
        allocate_task(free_node, base_tag + 1)
        flag = False
    else:
        flag = recv_task(base_tag + 1)

    if flag:
        # For ImageNet, use pipeline loader
        if config.dataset == "imagenet":
            trainset = MyImagenet(root='./data', train=True, transform=transform, rank=config.rank)
            trainset.creat_pipe(trainloader)
            trainloader = trainset

        net.eval()  # Target classifier is fixed
        for _ in range(15):  # Train backdoor generator for several epochs
            tot_loss, length = 0, 0
            backdoor_model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)
                # Generate adversarial noise using backdoor model
                noise = backdoor_model(inputs) * config.atk_eps
                noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
                perturbed_inputs = torch.clamp(inputs + noise, -1.0, 1.0)
                # Force labels to target class
                labels.fill_(config.target_label)
                outputs = net(perturbed_inputs)
                loss = criterion(outputs, labels)
                backdoor_model_optimizer.zero_grad()
                loss.backward()
                backdoor_model_optimizer.step()
                tot_loss += loss.item()
                length += len(inputs)
                if i > TRAIN_EPOCH:
                    break

        # Restore classifier to training mode, freeze backdoor model
        net.train()
        backdoor_model.eval()

        # Send trained backdoor model weights back to server
        torch.cuda.empty_cache()
        comm.send((client_idx, dict_to_cpu(backdoor_model.state_dict())), 0, 2)
        comm.send(config.rank, 0, 0)


def local_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                defender_method=DEFENDER_TWO_STAGE_WITH_L2, batch=0, participant_rate=1,
                num_clients=0, base_tag=0, client_idx=None, local_model_dicts=None,
                backdoor_model_dicts=None, experiment_id=None, config: Config=None):
    # Initialize optimizer and loss function for worker clients
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

    # Define number of local training epochs (fixed to 1 here)
    local_epochs = 1

    # Decide which training role this client takes
    if backdoor:
        # Step 1: Always train the auxiliary backdoor model first
        backdoor_model_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                             defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                             local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                             optimizer, backdoor_model_optimizer, local_epochs, config)
        # Step 2: If client is designated as defender, run defense training
        if defender and (defender_method in [DEFENDER_TWO_STAGE_WITH_L2, DEFENDER_ONE_STAGE,
                                             DEFENDER_TWO_STAGE_NOL2, DEFENDER_TWO_STAGE_WEIGHTL2]):
            defender_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                           defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                           local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                           optimizer, backdoor_model_optimizer, local_epochs, config)
        else:
            # Otherwise, act as attacker
            attacker_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                           defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                           local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                           optimizer, backdoor_model_optimizer, local_epochs, config)
    else:
        # Benign client: perform standard supervised training
        normal_train(net, trainloader, backdoor, testloader, backdoor_model, defender, param_ratios,
                     defender_method, batch, participant_rate, num_clients, base_tag, client_idx,
                     local_model_dicts, backdoor_model_dicts, experiment_id, criterion,
                     optimizer, backdoor_model_optimizer, local_epochs, config)


