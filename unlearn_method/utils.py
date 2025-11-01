from config import Config
import torch

class Arguments():
    def __init__(self,config:Config):
        #Federated Learning Settings
        self.N_total_client = config.num_clients
        self.N_client = config.num_clients
        self.data_name = config.dataset# purchase, cifar10, mnist, adult
        self.global_epoch = 2
        
        self.local_epoch = 10
        
        
        #Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        
        
        #Federated Unlearning Settings
        self.unlearn_interval= 1#Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 4 #If want to forget, change None to the client index
        
                                #If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = False#If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.
        
        self.if_unlearning = True#If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training
        
        self.forget_local_epoch_ratio = 0.5 #When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
                                            #forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False
