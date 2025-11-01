import torch
import pickle
import timm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import copy
from dataset import MyImagenet,CIFAR10
from model import MNISTAutoencoder, resnet18,Net,UNet
from config import Config
from unlearn_method.distillation import distillation_unlearn
from unlearn_method.sga import sga
from unlearn_method.retrain import retrain
from unlearn_method.federaser import fedEraser
from unlearn_method.robustFu import robustFu
from unlearn_method.pgd import pgd
from dataset import MNIST

def unlearn(config:Config):
    if config.method == "distillation":
        model = distillation_unlearn(trainset,config.global_nets,config.local_nets,config.unlearn_target,config)
    elif config.method == "sga":
        sga(config)
    elif config.method == "pgd":
        pgd(config)
    elif config.method == "retrain":
        retrain(config)
    elif config.method == "fedEraser":
        fedEraser(config)
    elif config.method == "robustFu":
        robustFu(config)
        
def load_model(config:Config):
    global_nets = []
    local_nets = []
    i = 1 if config.method == "distillation" or config.method == "fedEraser" else config.round
    while i <= config.round:
        flag = False
        if os.path.exists(f"{config.save_dir}/global_net_model_{config.name}_round_{i}.pkl"):
            with open(f"{config.save_dir}/global_net_model_{config.name}_round_{i}.pkl","rb") as f:
                global_nets.append(pickle.load(f))
            flag = True
        if os.path.exists(f"{config.save_dir}/local_net_model_{config.name}_round_{i}.pkl"):
            with open(f"{config.save_dir}/local_net_model_{config.name}_round_{i}.pkl","rb") as f:
                local_nets.append(pickle.load(f))
            flag = True
        if os.path.exists(f"{config.save_dir}/backdoor_net_model_{config.name}_round_{i}.pkl"):
            with open(f"{config.save_dir}/backdoor_net_model_{config.name}_round_{i}.pkl","rb") as f:
                backdoor_dict = pickle.load(f)
            flag = True
        if not flag:
            break
        i += 1
    
    if config.attack_method == "iba" and config.method != "retrain":
        config.backdoor_model.load_state_dict(backdoor_dict)
        config.backdoor_model.eval()
    
    config.global_nets = global_nets
    config.local_nets = local_nets
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'imagenet'] ,help='Dataset for training')
    parser.add_argument('--name', type=str,default="",help='train name')
    parser.add_argument('--save_dir', type=str,default="checkpoints",help='save_dir')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size.') 
    parser.add_argument('--round', type=int, default=100, help='final training round') 
    parser.add_argument('--unlearn_target', type=int, default=4, help='') 
    parser.add_argument('--method', type=str,choices=['sga','distillation',"retrain",'fedEraser','robustFu',"pgd"],default="retrain", help='unlearn method')

    parser.add_argument('--attackMethod', type=str, choices=['dba', 'iba'])
    parser.add_argument('--gaussian_noise', action = 'store_true', help='Apply gaussian noise.')
    parser.add_argument('--geometric_median', action = 'store_true', help='Apply geometric median.')
    parser.add_argument('--o_distance_filter', action = 'store_true', help='Apply o_distance_filter.')
    parser.add_argument('--c_distance_filter', action = 'store_true', help='Apply c_distance_filter.')
    parser.add_argument('--flame', action = 'store_true', help='Apply Flame.')
    parser.add_argument('--multi_krum', action = 'store_true', help='Apply multi_krum.')
    parser.add_argument('--median', action = 'store_true', help='Apply median.')
    parser.add_argument('--AlignIns', action = 'store_true', help='Apply AlignIns.')
    parser.add_argument('--Fuba', action = 'store_true', help='Apply Fuba from ******.')
    parser.add_argument('--Indicator', action = 'store_true', help='Apply Indicator from Backdoor-indicator-defense.')
    parser.add_argument('--IBMFL', action = 'store_true', help='Apply Indicator from Identify Backdoored Model in Federated Learning via Individual Unlearning.')
    parser.add_argument('--Enforce', action = 'store_true', help='Apply Enforce defence from Potential defence.')
    parser.add_argument('--Topk', action = 'store_true', help='Apply Topk defence from Potential defence.')
    parser.add_argument('--target_label', type=int, default=8,help='Target label index')
    parser.add_argument('--nb_attack', type=int,default=4, help='Number of malicious clients.')
    parser.add_argument('--nb_defence', type=int, default=1,help='Number of malicious defence clients.')
    parser.add_argument('--nb_clients', type=int, default=20,help='Number of all clients.')
    parser.add_argument('--forgot_client_idx', type=int, default=4,help='')
    parser.add_argument('--noise_thread', type=float, default=0.04, help='Noise threadhold for iba attack.')
    parser.add_argument('--atk_eps', type=float, default=1, help='atk eps for iba attack.')
    parser.add_argument('--communication_rounds', type=int, default=100, help='Number of FL rounds.')
    parser.add_argument('--train_epoch', type=int, default=100, help='Number of batch a client train each fl round.')
    parser.add_argument('--participant_rate', type=float, default=0.4, help='Client participant rate for each fl round')
    parser.add_argument('--warm_up', type=int, default=5, help='Number of warm up round.')
    parser.add_argument('--test_batch_size', type=int, default=32, help='test batch size.')
    parser.add_argument('--non_iid', action="store_true", help='')
    parser.add_argument('--non_iid_type', type=str,choices=['lognormal','Dirichlet','concept_shift'], help='')
    parser.add_argument('--alpha', type=float, default=1, help='alpha for non-iid Dirichlet setting')
    parser.add_argument('--sigma', type=float, default=1, help='sigma for non-iid lognormal setting')
    parser.add_argument('--shift_ratio', type=float, default=0.4, help='non-iid concept shift setting')
    parser.add_argument('--data_path', '--data_path', type=str, default=os.getcwd(),
                        help="Path where dataset is stored or will be downloaded.")
    parser.add_argument('--save', action='store_true', help='Save results (.pth) after FL training and after defense.'
                                                            ' Otherwise do not mention.')
    parser.add_argument('--n_gpu', type=int, help='number of gpus availablo')
    parser.add_argument('--no_detail_test', action="store_false", help='Apply detail test')
    parser.add_argument('--no_global_test', action="store_false", help='Apply detail test')

    args = parser.parse_args()
    config = Config(args,1,1)
    

    if config.dataset.upper() == "MNIST":
        if config.attack_method == "iba":
            backdoor_model = MNISTAutoencoder().to(config.device)
        else:
            backdoor_model = None
        transform = transforms.Compose(
                [transforms.ToTensor()])
        testset = MNIST(root='./data', train=False,
                                                download=True, transform=transform)
        trainset = MNIST(root='./data', train=True,
                                                download=True, transform=transform)

    elif config.dataset.upper() == "CIFAR10":
        transform = transforms.Compose(
                [transforms.ToTensor()])
        testset = CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        backdoor_model = UNet(3).to(config.device)

    elif config.dataset.upper() == "IMAGENET":
        backdoor_model = UNet(3).to(config.device)
        model = lambda:timm.create_model("hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=True)
        transform_t = timm.data.create_transform(**data_config, is_training=False)
        trainset = MyImagenet(root='./data', train=True,
                                            transform=transform,rank=1)
        testset = MyImagenet(root='./data', train=False,
                                            transform=transform_t,rank=1)
    else:
        raise ValueError()
    
    config.backdoor_model = backdoor_model
    config.trainset = trainset
    config.testset = testset
    config.transform = transform
    load_model(config)
    unlearn(config)
