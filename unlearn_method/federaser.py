import torch
from config import Config
from .FedEraser.Fed_Unlearn_base import unlearning
from .utils import Arguments

def fedEraser(config:Config):
    trainloaders = []
    subset_length = len(config.trainset) // config.num_clients
    remainder = len(config.trainset) % config.num_clients
    lengths = [subset_length + 1] * remainder + [subset_length] * (config.num_clients - remainder)
    torch.manual_seed(522)
    trainset_split = torch.utils.data.random_split(config.trainset, lengths)
    for subset in trainset_split:
        if config.dataset == 'imagenet':
            trainloaders.append(subset)
        else:
            trainloaders.append(torch.utils.data.DataLoader(subset, batch_size=config.batch_size,
                                                            shuffle=True, num_workers=0,drop_last=True))

    testloader = torch.utils.data.DataLoader(config.testset, batch_size=32,
                                            shuffle=True, num_workers=12)
    Old_GMS = []
    Old_CMS = []
    for i in range(len(config.global_nets)):
        Old_GMS.append(config.global_nets[i])
    for i in range(len(config.local_nets)):
        Old_CMS.append([])
        for j in range(len(config.local_nets[i])):
            Old_CMS[-1].append(config.local_nets[i][j])
    unlearning(Old_GMS,Old_CMS,trainloaders,testloader,Arguments(config),config)
        