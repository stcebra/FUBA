import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyImagenet
from utils.comm_utils import attack, test, save_models
from config import Config
from utils.dba_attack_utils import add_backdoor_all

def unlearning_loss(model, global_params, fisher_matrix, inputs, targets, lambda_val, config:Config):
    # Compute cross-entropy loss on current inputs
    ce_loss = cross_entropy_loss(model(inputs), targets)

    # Fisher-based regularization term: penalizes deviation from global_params
    regularization_term = 0
    for name, param in model.named_parameters():
        regularization_term += (fisher_matrix[name] * (param - global_params[name].to(config.device)) ** 2).sum()

    # Debug print: CE loss and regularization
    print(ce_loss, lambda_val / 2 * regularization_term)

    # Return adjusted unlearning loss (subtract Fisher penalty)
    return ce_loss - (lambda_val / 2) * regularization_term

    
def cross_entropy_loss(output, target):
    # Wrapper for standard cross-entropy loss
    return nn.CrossEntropyLoss()(output, target)


def sga_unlearning(model, global_params, data_loader, fisher_matrix, lambda_val, learning_rate, epochs, config:Config):
    # Special case: wrap data loader for ImageNet
    if config.dataset == "Imagenet":
        trainset = MyImagenet(root='./data', train=True, transform=config.transform, rank=1)
        trainset.creat_pipe(data_loader)
        data_loader = trainset

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Add perturbations depending on attack method
            if config.attack_method == "iba":
                noise = config.backdoor_model(inputs) * config.atk_eps
                noise = torch.clamp(noise, -config.noise_thread, config.noise_thread)
                perturbed_inputs = torch.clamp(inputs + noise, -1.0, 1.0)
            elif config.attack_method == "dba":
                perturbed_inputs = add_backdoor_all(inputs)
            else:
                perturbed_inputs = inputs

            # Compute unlearning loss
            loss = unlearning_loss(model, global_params, fisher_matrix, perturbed_inputs, targets, lambda_val, config)

            # Early stop if loss becomes unstable
            if loss.item() > 3:
                return model

            # Backpropagation and manual update of parameters
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param += learning_rate * param.grad

    return model


def fisher_information(model, data_loader, config:Config):
    # Special case: wrap data loader for ImageNet
    if config.dataset == "Imagenet":
        trainset = MyImagenet(root='./data', train=True, transform=config.transform, rank=1)
        trainset.creat_pipe(data_loader)
        data_loader = trainset

    # Initialize Fisher matrix (same shape as model params)
    fisher_matrix = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    model.eval()
    for t in data_loader:
        if t is None:
            break
        else:
            inputs, targets = t

        model.zero_grad()
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        # Forward pass and compute gradients
        outputs = model(inputs)
        loss = cross_entropy_loss(outputs, targets)
        loss.backward()

        # Accumulate squared gradients into Fisher matrix
        for name, param in model.named_parameters():
            fisher_matrix[name] += param.grad.data ** 2

    # Normalize Fisher matrix by dataset size
    fisher_matrix = {name: param / len(data_loader) for name, param in fisher_matrix.items()}
    return fisher_matrix


def sga(config:Config):
    # Select correct attack function depending on method
    if config.attack_method == "dba":
        from utils.comm_utils import attack_dba as attack
    else:
        from utils.comm_utils import attack

    # Load last global model for unlearning
    unlearn_net = config.creat_cls_net().to(config.device)
    unlearn_net.load_state_dict(config.global_nets[-1])

    # Split dataset into client subsets
    subset_length = len(config.trainset) // config.num_clients
    remainder = len(config.trainset) % config.num_clients
    lengths = [subset_length + 1] * remainder + [subset_length] * (config.num_clients - remainder)

    torch.manual_seed(522)
    trainset_split = torch.utils.data.random_split(config.trainset, lengths)

    # Build data loaders for each client
    trainloaders = []
    for subset in trainset_split:
        if config.dataset == 'Imagenet':
            trainloaders.append(subset)
        else:
            trainloaders.append(torch.utils.data.DataLoader(subset, batch_size=config.batch_size,
                                                            shuffle=True, num_workers=12))

    # Compute Fisher information for forgotten client
    fisher_matrix = fisher_information(unlearn_net, trainloaders[config.forgot_client], config)

    # Hyperparameters
    lambda_val = 100
    learning_rate = 0.001
    epochs = 9

    # Test loader for evaluation
    testloader = torch.utils.data.DataLoader(config.testset, batch_size=config.batch_size,
                                             shuffle=False, num_workers=12)

    # Evaluate before unlearning
    print("Before unlearn")
    print("Accuracy:", test(unlearn_net, testloader, config.device), "%")
    print("ASR:", attack(model=unlearn_net, loader=testloader, atkmodel=config.backdoor_model,
                         ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                         target_label=config.target_label, device=config.device,
                         backdoor_func=add_backdoor_all), "%")

    # Perform SGA unlearning on forgotten client
    updated_model = sga_unlearning(unlearn_net, config.global_nets[-1], trainloaders[config.forgot_client],
                                   fisher_matrix, lambda_val, learning_rate, epochs, config)

    # Evaluate after unlearning
    print("After unlearn")
    print("Accuracy:", test(updated_model, testloader, config.device), "%")
    print("ASR:", attack(model=unlearn_net, loader=testloader, atkmodel=config.backdoor_model,
                         ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                         target_label=config.target_label, device=config.device,
                         backdoor_func=add_backdoor_all), "%")

    # Save unlearned model
    save_models(updated_model.state_dict(), None,
                None if config.attack_method == "dba" else config.backdoor_model,
                0, "sga_" + config.name)

    return updated_model