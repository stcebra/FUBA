import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils.comm_utils import attack
from utils.comm_utils import l2_distance, attack, test,get_backdoored_dataset ,get_clean_dataset, save_models, split_non_iid_dirichlet,find_max_idx_within_limit,federated_averaging_net,geometric_median,compute_parameter_ratio_difference,adjust_updates_based_on_ratio,dict_to_cpu,dict_to_device,lognormal_split,split_non_iid_concept
from config import Config
from utils.dba_attack_utils import add_backdoor_all

def train_pattern(trainloader, backdoor_model, net, config: Config):
    """
    Train a backdoor pattern generator model against a fixed classifier.
    
    Args:
        trainloader (DataLoader): Training data loader.
        backdoor_model (nn.Module): Backdoor pattern generator network.
        net (nn.Module): Target classifier network.
        config (Config): Experiment configuration object.
    """
    backdoor_model.train()
    net.eval()
    criterion = nn.CrossEntropyLoss()
    backdoor_model_optimizer = optim.Adam(backdoor_model.parameters())

    for _ in range(10):
        tot_loss = 0
        length = 0
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(config.device, non_blocking=True), data[1].to(config.device, non_blocking=True)

            # Generate adversarial noise
            noise = backdoor_model(inputs) * config.atk_eps
            noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)

            perturbed_inputs = torch.clamp(inputs + noise, -1.0, 1.0)

            # Replace labels with target backdoor label
            labels.fill_(config.target_label)

            # Compute loss on perturbed inputs
            outputs = net(perturbed_inputs)
            loss = criterion(outputs, labels)

            # Optimize backdoor model
            backdoor_model_optimizer.zero_grad()
            loss.backward()
            backdoor_model_optimizer.step()

            tot_loss += loss.item()
            length += len(inputs)

        print(tot_loss / length)

    backdoor_model.eval()
    net.train()


def distillation_unlearn(trainset, global_dicts, clients_dicts, idx, config: Config):
    """
    Perform distillation-based unlearning on a selected client model.

    Args:
        trainset (Dataset): Training dataset.
        global_dicts (list[dict]): List of global model checkpoints across rounds.
        clients_dicts (list[list[dict]]): List of per-client model checkpoints.
        idx (int): Index of the client to unlearn.
        config (Config): Experiment configuration object.

    Returns:
        dict: State dict of the unlearned model.
    """
    # Choose attack method
    if config.attack_method == "dba":
        from utils.comm_utils import attack_dba as attack
    else:
        from utils.comm_utils import attack

    cou = -1  # Use last global checkpoint
    testloader = torch.utils.data.DataLoader(
        config.testset, batch_size=config.batch_size,
        shuffle=False, num_workers=12
    )

    # Create new global model dictionary by removing influence of selected client
    D = copy.deepcopy(global_dicts[cou])
    for k in D.keys():
        local_updates = [unlearn_dicts[idx][k].float() for unlearn_dicts in clients_dicts]
        global_updates = [global_dict[k].float() for global_dict in global_dicts[:cou]]
        avg_update = torch.stack(local_updates, 0).sum(0) / 20 - torch.stack(global_updates, 0).sum(0) / 20
        D[k] = D[k] - avg_update

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size,
        shuffle=True, num_workers=4
    )

    # Initialize student model with modified global weights
    net = config.creat_cls_net().to(config.device)
    net.load_state_dict(D)

    # Evaluate student before unlearning
    print(f"""========================
          To unlearn Model, ACC: {test(net,testloader,config.device)}%, 
          ASR: {attack(model=net, loader=testloader, atkmodel=config.backdoor_model,
                        ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                        target_label=config.target_label, device=config.device,
                        backdoor_func=add_backdoor_all)}%
          ========================""")

    # Teacher model is the original global checkpoint
    teacher_net = config.creat_cls_net().to(config.device)
    teacher_net.load_state_dict(global_dicts[cou])

    # Evaluate teacher
    print(f"""========================
          Teacher Model, ACC: {test(teacher_net,testloader,config.device)}%, 
          ASR: {attack(model=teacher_net, loader=testloader, atkmodel=config.backdoor_model,
                        ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                        target_label=config.target_label, device=config.device,
                        backdoor_func=add_backdoor_all)}%
          ========================""")

    # If IBA attack: retrain backdoor model patterns
    if config.attack_method == "iba":
        #train_pattern(trainloader, config.backdoor_model, net, config)
        print(attack(model=net, loader=testloader, atkmodel=config.backdoor_model,
                        ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                        target_label=config.target_label, device=config.device,
                        backdoor_func=add_backdoor_all))

    # Distillation setup
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    temperature = 30
    soft_criterion = nn.KLDivLoss(reduction='batchmean')
    criterion = nn.CrossEntropyLoss()

    # Distillation training loop
    for e in range(1):
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(config.device), data[1].to(config.device)

            # Apply universal backdoor transformation
            inputs = add_backdoor_all(inputs)

            optimizer.zero_grad()

            # Teacher predictions (frozen)
            with torch.no_grad():
                teacher_outputs = teacher_net(inputs)

            # Student predictions
            student_outputs = net(inputs)

            # Hard label loss
            notloss = criterion(student_outputs, labels)
            _, predicted = torch.max(student_outputs.data, 1)

            # Soft label distillation loss
            soft_loss = soft_criterion(
                torch.log_softmax(student_outputs / temperature, dim=1),
                torch.softmax(teacher_outputs / temperature, dim=1)
            )

            loss = soft_loss
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 20 == 0:
                print(f"acc={correct/total:06.3f} loss={notloss:06.3f} softloss={loss:06.3f}", end="\r")

        # Evaluate model after each epoch
        print(f"""========================
          To unlearn Model, ACC: {test(net,testloader,config.device)}%, 
          ASR: {attack(model=net, loader=testloader, atkmodel=config.backdoor_model,
                        ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                        target_label=config.target_label, device=config.device,
                        backdoor_func=add_backdoor_all)}%
          ========================""")
        
        if config.attack_method == "iba":
            train_pattern(trainloader, config.backdoor_model, net, config)
            
        print(f"""========================
          To unlearn Model, ACC: {test(net,testloader,config.device)}%, 
          ASR: {attack(model=net, loader=testloader, atkmodel=config.backdoor_model,
                        ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                        target_label=config.target_label, device=config.device,
                        backdoor_func=add_backdoor_all)}%
          ========================""")

    return net.state_dict()
