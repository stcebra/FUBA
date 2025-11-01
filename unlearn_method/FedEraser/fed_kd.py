
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time 

#ourself libs
from model_initiation import model_init
from data_preprocess import data_init, data_init_with_shadow
from FL_base import global_train_once
from FL_base import fedavg
from FL_base import test

from FL_base import FL_Train, FL_Retrain
from Fed_Unlearn_base import unlearning, unlearning_without_cali, federated_learning_unlearning
from membership_inference import train_attack_model, attack
class Arguments():
    def __init__(self):
        #Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10
        self.data_name = 'mnist'# purchase, cifar10, mnist, adult
        self.global_epoch = 20
        
        self.local_epoch = 10
        
        
        #Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True

def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    torch.manual_seed(FL_params.seed)
    init_global_model = model_init(FL_params.data_name)
    client_loaders, test_loader = data_init(FL_params)
    print(5*"#"+"  Federated Learning Start"+5*"#")
    std_time = time.time()
    old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_learn = (std_time - end_time)
    print(5*"#"+"  Federated Learning End"+5*"#")

