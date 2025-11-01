import torch
from model import resnet18
from model import Net
import timm
from argparse import Namespace
import sys

class Config:
    def __init__(self, args, rank, size):
        # MPI rank and world size
        self.rank = rank
        self.size = size

        # General experiment settings
        self.dataset = args.dataset
        self.name = args.name
        try:
            self.pretrained_model = args.pretrained_model
            self.test_only = args.test_only
        except:
            self.pretrained_model = None
            self.test_only = False

        # Attack-related settings
        self.attack_method = args.attackMethod
        self.target_label = args.target_label
        self.nb_attack = args.nb_attack
        self.nb_defence = args.nb_defence
        self.forgot_client = args.forgot_client_idx

        # Backdoor attack parameters
        self.noise_thread = args.noise_thread
        self.atk_eps = args.atk_eps

        # Federated learning parameters
        self.num_clients = args.nb_clients
        self.communication_rounds = args.communication_rounds
        self.participant_rate = args.participant_rate
        self.warm_up = args.warm_up
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size

        # Non-IID data distribution parameters
        self.non_iid = args.non_iid
        self.non_iid_type = args.non_iid_type
        self.shift_ration = args.shift_ratio
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.param_ratios = None

        # Hardware settings
        self.n_gpu = args.n_gpu
        self.data_path = args.data_path
        self.save_results = args.save

        # Testing settings
        self.apply_detail_test = args.no_detail_test
        self.apply_global_test = args.no_global_test

        # Defense method toggles
        self.apply_gaussian_noise = args.gaussian_noise
        self.apply_geometric_median = args.geometric_median
        self.apply_o_distance_filter = args.o_distance_filter
        self.apply_c_distance_filter = args.c_distance_filter
        self.apply_flame = args.flame
        self.apply_multy_krum = args.multi_krum
        self.apply_median = args.median
        self.apply_alignins = args.AlignIns
        self.apply_fuba = args.Fuba
        self.apply_indicator = args.Indicator
        self.apply_ibmfl = args.IBMFL
        try:
            self.apply_fldetector = args.flDetector
        except:
            pass
        try:
            self.apply_enforce = args.Enforce
            self.apply_topk = args.Topk
        except:
            pass

        # Arguments for unlearning-specific tasks
        try:
            self.unlearn_target = args.unlearn_target
            self.method = args.method
            self.round = args.round
            self.save_dir = args.save_dir

            # Initialize placeholders for models and datasets
            self.global_nets = None
            self.backdoor_model = None
            self.local_nets = None
            self.trainset = None
            self.testset = None
            self.transform = None
        except:
            pass
        
        # Ensure at most one defense strategy is applied at a time
        flags = [
            args.gaussian_noise,
            args.geometric_median,
            args.o_distance_filter,
            args.c_distance_filter,
            args.flame,
            args.multi_krum,
            args.median,
            args.AlignIns,
            args.Fuba,
            args.Indicator,
            args.IBMFL
        ]
        assert sum(bool(f) for f in flags) <= 1

        # Assign device to each process
        self.device = torch.device(
            f"cuda:{rank % args.n_gpu}"
            if torch.cuda.is_available() and args.n_gpu and args.n_gpu > 0
            else "cpu"
        )

        # Special rule: rank 0 (server) and rank == size (evaluator) always use GPU 0
        if rank == 0 or rank == size:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() and args.n_gpu and args.n_gpu > 0 else "cpu"
            )

        # Default control flags
        self.grad_simulate_train = True   # Whether to simulate gradient-based training
        self.simulate_train = True        # Whether to simulate client local training
        self.clients_ratial = 1           # Ratio multiplier for number of clients
        self.step_size = 0.1              # Step size for adversarial scaling
        self.num_clients *= self.clients_ratial

        # Model creation functions for different datasets
        if self.dataset == "mnist":
            self.creat_cls_net = lambda: Net()
        elif self.dataset == "cifar10":
            self.creat_cls_net = lambda: resnet18()
        elif self.dataset == "imagenet":
            self.creat_cls_net = lambda: timm.create_model(
                "hf_hub:timm/mobilenetv4_conv_medium.e500_r256_in1k",
                pretrained=True
            )

        # Defense integration: FLAME
        if self.apply_flame:
            print("Defence frame is applied")
            sys.path.append("defence/FLAME/utils")
            from defense import flame, get_update
            self.flame = flame
            self.get_update = get_update  # NOTE: might be a typo (comma instead of dot)

        # Defense integration: AlignIns
        if self.apply_alignins:
            from defence.AlignIns.src.aggregation import Aggregation
            self.Aggregation = Aggregation

        # Defense integration: Backdoor Indicator
        if self.apply_indicator:
            from defence.Backdoor_indicator_defense.participants.servers.IndicatorServer import IndicatorServer
            from defence.Backdoor_indicator_defense.dataloader.WMFLDataloader import WMFLDataloader
            import yaml
            with open(f"./defence/Backdoor_indicator_defense/utils/yamls/indicator/params_dba_Indicator.yaml", "r") as f:
                params_loaded = yaml.safe_load(f)
            params_loaded["ood_data_source"] = "NOISE"
            params_loaded["model_type"] = "ResNet"
            self.indicator_param = params_loaded
            self.indicator_loader = WMFLDataloader(params_loaded)
            self.IndicatorServer = IndicatorServer

        # Defense integration: IBMFL (MASA)
        if self.apply_ibmfl:
            from defence.MASA.src.aggregation import Aggregation
            from torch.nn.utils import parameters_to_vector
            IBMFL_args = Namespace(
                data='cifar10',
                num_agents=20,
                agent_frac=1.0,
                num_corrupt=2,
                rounds=100,
                local_ep=2,
                attacker_local_ep=2,
                bs=64,
                client_lr=0.1,
                malicious_client_lr=0.1,
                server_lr=0.5,
                target_class=7,
                poison_frac=0.5,
                pattern_type='plus',
                theta=8,
                theta_ld=10,
                snap=1,
                device=self.device,
                num_workers=0,
                dense_ratio=0.4,
                anneal_factor=0.0001,
                non_iid=False,
                debug=False,
                alpha=0.5,
                attack='badnet',
                aggr='masa',
                lr_decay=0.99,
                momentum=0.0,
                mask_init='ERK',
                dis_check_gradient=False,
                wd=1e-4,
                same_mask=1,
                exp_name_extra='',
                super_power=False,
                clean=False,
                fusion_lambda=0.4,
                filter_delta=2.0,
                unlearn_lr=0.0001,
                unlearn_moment=0.9
            )

            self.Aggregation = Aggregation
            self.IBMFL_args = IBMFL_args
            self.aggrator = None
