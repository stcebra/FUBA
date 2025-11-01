import copy
import torch
from torch import nn
from config import Config
from utils.comm_utils import l2_distance, attack, test,get_backdoored_dataset ,get_clean_dataset, save_models, split_non_iid_dirichlet,find_max_idx_within_limit,federated_averaging_net,geometric_median,compute_parameter_ratio_difference,adjust_updates_based_on_ratio,dict_to_cpu,dict_to_device,lognormal_split,split_non_iid_concept
from config import Config
from utils.dba_attack_utils import add_backdoor_all
from .retrain import retrain


def derated_unlearn_all_rounds(config: Config):
    """
    Perform derated unlearning across all communication rounds by excluding
    the target client and averaging the remaining client updates.

    Args:
        config (Config): Experiment configuration object.

    Returns:
        dict: State dict of the unlearned global model (last round).
    """
    num_rounds = len(config.local_nets)
    num_clients = len(config.local_nets[0])

    unlearned_global_nets = []

    for round_idx in range(num_rounds):
        client_models = config.local_nets[round_idx]

        # Exclude the target client
        selected_clients = [i for i in range(num_clients) if i != config.unlearn_target]

        with torch.no_grad():
            sum_vec = None

            # Aggregate model vectors from selected clients
            for idx in selected_clients:
                model = config.creat_cls_net()
                model.load_state_dict(client_models[idx])

                vec = nn.utils.parameters_to_vector(model.parameters())

                if sum_vec is None:
                    sum_vec = vec.clone()
                else:
                    sum_vec += vec

            # Average the aggregated vectors
            sum_vec /= len(selected_clients)

            # Load averaged vector into a new model
            unlearned_model = config.creat_cls_net()
            nn.utils.vector_to_parameters(sum_vec, unlearned_model.parameters())

        unlearned_global_nets.append(unlearned_model.state_dict())

    # Return final round unlearned global model
    return unlearned_global_nets[-1]


def pgd(config: Config):
    """
    Perform PGD-style retraining after unlearning, then evaluate accuracy and ASR.

    Args:
        config (Config): Experiment configuration object.
    """
    # Select appropriate attack function
    if config.attack_method == "dba":
        from utils.comm_utils import attack_dba as attack
    else:
        from utils.comm_utils import attack

    # Build unlearned model
    net = derated_unlearn_all_rounds(config)
    model = config.creat_cls_net().to(config.device)
    model.load_state_dict(net)

    # Prepare test loader
    testloader = torch.utils.data.DataLoader(
        config.testset, batch_size=config.batch_size,
        shuffle=False, num_workers=12
    )

    # Reset communication rounds and set pretrained model
    config.communication_rounds = 20
    config.pretrained_model = model

    # Retrain model and backdoor model
    model, backdoored_model = retrain(config)
    config.backdoor_model = backdoored_model

    # Evaluate and print results
    print(f"""========================
          To unlearn Model, ACC: {test(model, testloader, config.device)}%, 
          ASR: {attack(model=model, loader=testloader, atkmodel=config.backdoor_model,
                        ATK_EPS=config.atk_eps, NOISE_THREAD=config.noise_thread,
                        target_label=config.target_label, device=config.device,
                        backdoor_func=add_backdoor_all)}%
          ========================""")
