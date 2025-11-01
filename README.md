# FUBA: Backdoor Federated Learning via Federated Unlearning

## Abstract

Federated unlearning (FU) methods enable participants in federated learning (FL) to exercise the “right to be forgotten (RTBF)” by removing their contributions from a collaboratively trained model. While existing studies on FU primarily focus on accelerating the unlearning process and improving the utility of the unlearned model, they often overlook the integrity and validity of unlearning requests themselves. This gap creates a vulnerability, as malicious attackers could exploit RTBF to launch adversarial attacks on the unlearned model. In this study, we propose a novel federated unlearning backdoor attack framework (FUBA), which leverages malicious unlearning requests to backdoor the unlearned model after FU. FUBA models this attack as an adversarial game between two types of adversarial clients: the Adv-attacker and the Adv-defender. Specifically, during the FL process, the Adv-attacker continuously injects backdoor triggers into the global model, while the Adv-defender acts to mitigate these injections. Then, in the FU phase, the Adv-defender activates the backdoor by submitting an unlearning request to remove its contribution. To enhance the effectiveness and stability of FUBA, the Adv-attacker incorporates an innovative gradient reweighting mechanism and an adaptive model update scaling algorithm for durable and stable trigger embedding. Meanwhile, the Adv-defender employs an advanced two-stage defense vector generation strategy and a model consistency loss to maintain behavioral consistency during training, ensuring reliable backdoor activation after unlearning. Extensive experimental results demonstrate that FUBA effectively and reliably backdoors the unlearned model after FU, regardless of the FU method applied. Furthermore, FUBA’s unique design ensures a high level of stealth, rendering existing backdoor defenses inadequate.

## Requirements

* Python 3.11
* PyTorch
* MPI4py
* torchvision
* mpi
* nvidia cuda dali

## Setup

1. Download the repository:

2. Install dependencies:
   - 2.1 install pytorch that suits your computer
   - 2.2 install mpich
      sudo apt-get install mpich
      MPICC=/usr/bin/mpicc pip3 install --no-binary=mpi4py mpi4py
   - 2.3 Install nvidia cuda dali
      Follow instructions in https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
   - 2.4 install other non-python dependency
      sudo apt update
      sudo apt install libpq-dev
   - 2.5 Install Other dependency
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

1. Train with MPI-based FL:  

   Sample command: Conduct FUBA on MNIST dataset with IBA as the backdoor trigger
   ```bash
   mpiexec -n 5 python3 main.py --dataset=mnist --name=test1 --no_global_test --no_detail_test --attackMethod=iba --n_gpu=1 --save
   ```
   **--dataset**
   Specifies the dataset to be used for training. Options: mnist, cifar10, imagenet. MANDATORY

   **--name**
   Assigns a custom name to this training run (e.g., test1). This name is used to organize output directories and saved files. MANDATORY

   **--attackMethod**
   Specifies the attack method to be used for training. Options: iba, dba. MANDATORY

   **--save**
   If included, this flag enables saving of model checkpoints and training results (e.g., .pkl files).

   **--n_gpu**
   Specifies the number of GPUs to use during training.


2. Execute federated unlearning backdoor attack:  

   Sample command: Conduct unlearning with Retrain from Scratch baseline
   ```bash
   python3 unlearn.py --name=test1 --dataset=mnist --method=retrain --attackMethod=iba --n_gpu=1
   ```
   **--dataset**
   Specifies the dataset to be used for training and unlearning. Must match the dataset used in iba_mpi.py. Options: mnist, cifar10, imagenet. MANDATORY

   **--name**
   Specifies the training name. Should be the same as the one used when running main.py. MANDATORY

   **--method**
   Specifies the unlearning method to be applied. Options: retrain, fedEraser, distillation， etc.  MANDATORY

   **--attackMethod**
   Specifies the attack method to be used for training. Options: iba, dba. MANDATORY

3. Test pretrained/trained model:  

   Sample command: Evaluate the performance after Retrain from Scratch baseline
   ```bash
   python3 main.py --attackMethod=iba --dataset=mnist  --n_gpu=1 --test_only --pretrained_model=checkpoints/global_net_model_retrain_test1_round_99.pkl
   ```
   Sample command: Validate the attack performance of FUBA on MNIST dataset with SGA-based FU method
   ```bash
   python3 main.py --attackMethod=iba --dataset=mnist  --n_gpu=1 --test_only --pretrained_model=saved_models/global_sga_iba_mnist.pkl
   ```
   **--dataset**
   Specifies the dataset to be used for training and unlearning. Must match the dataset used in main.py. Options: mnist, cifar10, imagenet. MANDATORY

   **--pretrained_model**
   Specifies the model path for testing.

   **--attackMethod**
   Specifies the attack method to be used for training. Options: iba, dba. MANDATORY

4. Train with MPI-based FL with defence method:
   
   Sample Command: Running FUBA with IBA as the Backdoor Trigger with flame defence
   ```bash
   mpiexec -n 5 python3 main.py --dataset=mnist --name=test1 --no_global_test --no_detail_test --attackMethod=iba --n_gpu=1 --save --flame
   ```
   Add independent flag to execute certain defence method. E.g., --flame --multi_krum --median --AlignIns , etc. Only one defence method can be applied.
