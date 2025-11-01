from mpi4py import MPI
from config import Config
from train_main import main
import os

comm = MPI.COMM_WORLD
SIZE = comm.Get_size() - 1
RANK = comm.Get_rank()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'imagenet'] ,help='Dataset for training')
    parser.add_argument('--attackMethod', type=str, choices=['dba', 'iba'])
    parser.add_argument('--name', type=str,default="",help='train name')
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
    parser.add_argument('--flDetector', action = 'store_true', help='Apply Fldetector.')
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
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size.') #12
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
    
    parser.add_argument('--pretrained_model', type=str,default=None,help='')
    parser.add_argument('--test_only', action="store_true", help='test pretrained model only')

    args = parser.parse_args()

    config = Config(args, RANK, SIZE)
    
    main(config)