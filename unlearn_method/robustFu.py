import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, TensorDataset
from config import Config
from utils.comm_utils import test
from utils.dba_attack_utils import add_backdoor_all

def merge_datasets(loader1, loader2,config):
    # Initialize lists to hold combined data and targets
    combined_data = []
    combined_targets = []

    # Extract data and targets from the first loader
    with torch.no_grad():
        for data, targets in loader1:
            data,targets = data.to(config.device), targets.to(config.device)
            combined_data.append(data)
            combined_targets.append(targets)

    # Extract data and targets from the second loader
    with torch.no_grad():
        for data, targets in loader2:
            combined_data.append(data)
            combined_targets.append(targets)

    # Concatenate all data and targets into single tensors
    combined_data = torch.cat(combined_data, dim=0)
    combined_targets = torch.cat(combined_targets, dim=0)

    return combined_data, combined_targets

def robustFu(config:Config):
    if config.attack_method == "dba":
        from utils.comm_utils import attack_dba as attack
    else:
        from utils.comm_utils import attack
    # Calculate the size of two subsets
    max = 0
    part1_size = len(config.testset) // 5  # 1/5 of the data
    part2_size = len(config.testset) - part1_size  # the remaining 4/5 of the data

    # Create two subsets
    testset_part1 = Subset(config.testset, list(range(part1_size)))  # First part: first 20%
    testset_part2 = Subset(config.testset, list(range(part1_size, len(config.testset))))  # Second part: remaining 80%

    # Optionally create DataLoaders
    testloader_part1 = DataLoader(testset_part1, batch_size=64, shuffle=False)
    testloader_part2 = DataLoader(testset_part2, batch_size=64, shuffle=False)

    # Example: print the length of the two parts
    #print(f"Length of Testset Part 1: {len(testset_part1)}")
    #print(f"Length of Testset Part 2: {len(testset_part2)}")


    # Create DataLoader for each client
    subset_length = len(config.trainset) // config.num_clients
    remainder = len(config.trainset) % config.num_clients
    lengths = [subset_length + 1] * remainder + [subset_length] * (config.num_clients - remainder)

    to_unlearn_model = config.creat_cls_net().to(config.device)
    to_unlearn_model.load_state_dict(config.global_nets[-1])
        
    client_trainloaders = []
    torch.manual_seed(522)
    trainset_split = torch.utils.data.random_split(config.trainset, lengths)
    for subset in trainset_split:
        valid_indices = [idx for idx in subset.indices
                    if config.trainset[idx][1] != 3]  # keep only labels != 3
        subset = torch.utils.data.Subset(config.trainset, valid_indices)
        if config.dataset == 'imagenet':
            client_trainloaders.append(subset)
        else:
            client_trainloaders.append(torch.utils.data.DataLoader(subset, batch_size=config.batch_size,
                                                        shuffle=True, num_workers=0,drop_last=True))
        
    # Test global model accuracy
    testloader = DataLoader(config.testset, batch_size=config.batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader_part2:
            images, labels = data[0].to(config.device), data[1].to(config.device)
            outputs = to_unlearn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the global model on the 10000 test images: {100 * correct / total}%')

    # Test the network and print per-class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader_part2:
            images, labels = data[0].to(config.device), data[1].to(config.device)
            outputs = to_unlearn_model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    #for i in range(10):
        #print(f'Accuracy of {i} : {100 * class_correct[i] / class_total[i]:.2f}%')

    print("robustFU")

    # Initialize global model
    global_model = config.creat_cls_net().to(config.device)

    max_batches_per_client = 10  # Maximum number of batches per client per round

    scores = [0] * config.num_clients  # Example scoring list for clients

    # Start federated learning
    num_rounds = 30  # Simulate 30 rounds of federated learning
    for round in range(num_rounds):
        local_models = [config.creat_cls_net().to(config.device) for _ in range(config.num_clients)]
        local_weights = []

        # Clients train on their local data
        for client_id, trainloader in enumerate(client_trainloaders):
            #print(f"Training client {client_id+1}")
            local_model = local_models[client_id]
            local_model.load_state_dict(global_model.state_dict())  # Synchronize global model parameters to local model
            optimizer = optim.SGD(local_model.parameters(), lr=0.1, momentum=0.9)
            #optimizer = optim.Adadelta(local_model.parameters(), lr=1.0)
            batch_count = 0
            print(trainloader.dataset)
            for _, data in enumerate(trainloader, 0):
                if batch_count >= max_batches_per_client:
                    break
                inputs, labels = data[0].to(config.device), data[1].to(config.device)
                #optimizer.zero_grad()
                #outputs = local_model(inputs)
                #loss = F.cross_entropy(outputs, labels)
                #loss.backward()
                #ptimizer.step()
                batch_count += 1
                criterion = torch.nn.CrossEntropyLoss()
                if config.attack_method == "iba": 
                    noise = config.backdoor_model(inputs).detach() * config.atk_eps
                    noise = torch.clamp(noise,-config.noise_thread,config.noise_thread)
                    perturbed_inputs = inputs.clone()
                    mask = (torch.rand(len(inputs)) < 0.4).to(inputs.device)
                    perturbed_inputs[mask] = torch.clamp(inputs[mask] + noise[mask], -1.0, 1.0)
                else:
                    mask = (torch.rand(len(inputs)) < 0.4).to(inputs.device)
                    perturbed_inputs = inputs.clone()
                    perturbed_inputs[mask] = add_backdoor_all(inputs[mask])
                labels = torch.where(mask, torch.tensor(config.target_label, device=inputs.device), labels)
                optimizer.zero_grad()
                backdoored_outputs = local_model(perturbed_inputs)
                backdoored_loss = criterion(backdoored_outputs, labels)
                backdoored_loss.backward()
                #adjust_updates_based_on_ratio(net, param_ratios)
                optimizer.step()
            local_weights.append(local_model.state_dict())

            # Initialize counters
            conflict_count = 0
            total_samples = 0

            mismatched_data = []
            mismatched_targets = []

            # Compare local vs. global predictions on part1
            with torch.no_grad():
                for data in testloader_part1:
                    images, labels = data[0].to(config.device), data[1].to(config.device)

                    outputs1 = local_model(images)
                    _, predicted1 = torch.max(outputs1.data, 1)

                    outputs2 = to_unlearn_model(images)
                    _, predicted2 = torch.max(outputs2.data, 1)

                    # Calculate conflicts
                    conflicts = torch.ne(predicted1, predicted2)
                    conflict_count += conflicts.sum().item()
                    total_samples += labels.size(0)

                    pred1 = outputs1.argmax(dim=1, keepdim=True)  # Predictions by model 1
                    pred2 = outputs2.argmax(dim=1, keepdim=True)  # Predictions by model 2
                    mismatches = pred1.ne(pred2).view(-1)

                    if conflict_count > 0:
                        if torch.any(mismatches):
                            mismatched_data.append(images[mismatches])
                            mismatched_targets.append(pred2[mismatches].view(-1))

                        # Combine all mismatched samples into single tensors
                        mismatched_data_1 = torch.cat(mismatched_data, dim=0)
                        mismatched_targets_1 = torch.cat(mismatched_targets, dim=0)

                        mismatched_dataset = TensorDataset(mismatched_data_1, mismatched_targets_1)

                        mismatched_loader = DataLoader(mismatched_dataset, batch_size=config.batch_size, shuffle=True)

                #print(f"Number of conflicting predictions: {conflict_count}")
                #print(f"Total number of samples tested: {total_samples}")

                scores[client_id] = conflict_count

                # Apply inverse weighting
                inverse_weights = []
                for j in range(len(scores)):
                    if scores[j] > 0:
                        inverse_weights.append(1.0 / scores[j])
                    # else:
                    #     inverse_weights.append(1.0)
                norm_inverse_weights = [w / sum(inverse_weights) for w in inverse_weights]

            add = 1
            if add == 1 and conflict_count > 0 and round > 4:
                # Merge original training loader with mismatched loader
                combined_data, combined_targets = merge_datasets(client_trainloaders[client_id], mismatched_loader,config)

                # Create a new DataLoader with merged dataset
                combined_dataset = torch.utils.data.TensorDataset(combined_data, combined_targets)
                combined_loader = DataLoader(dataset=combined_dataset, batch_size=config.batch_size, shuffle=True)

                client_trainloaders[client_id] = combined_loader
        
        #print(scores)
        #print(norm_inverse_weights)

        fedavg = 0
        if fedavg == 1:
            # Update global model using simple averaging
            global_weights = dict()
            for key in local_weights[0]:
                global_weights[key] = torch.stack([local_weights[i][key] for i in range(config.num_clients)], 0).mean(0)
            global_model.load_state_dict(global_weights)

        if fedavg == 0:
            # Update global model using weighted averaging
            global_weights = dict()
            total_weight = sum(norm_inverse_weights)  # Sum of weights

            for key in local_weights[0]:
                # Weighted average for each parameter
                weighted_sum = sum(local_weights[i][key] * norm_inverse_weights[i] for i in range(config.num_clients))
                global_weights[key] = weighted_sum / total_weight

            global_model.load_state_dict(global_weights)

        # Evaluate global model accuracy
        testloader = DataLoader(config.testset, batch_size=config.batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(config.device), data[1].to(config.device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Evaluate per-class accuracy
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(config.device), data[1].to(config.device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        #for i in range(10):
        #    print(f'Accuracy of {i} : {100 * class_correct[i] / class_total[i]:.2f}%')

        acc = 100 * class_correct[i] / class_total[i]
        if acc > max:
            max = acc

        print(f'Accuracy of the global model on the 10000 test images: {100 * correct / total}%')

    ACC = test(global_model,testloader,config.device)
    ASR = attack(model=global_model, loader=testloader, atkmodel=config.backdoor_model,
                                 ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread, target_label=config.target_label, device=config.device,backdoor_func = add_backdoor_all)
    print(f"Final global acc {ACC}% asr {ASR}%")

