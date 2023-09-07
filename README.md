# Towards Unified Defense
Official repository for the Project: Towards Unified Defense: Adversarial Training, Watermarking, and Privacy Preservation, completed as part of MSc Individual Research @ ICL

## For training model in centralised and federated learning setting

### For Centralised: 
- run **centralised/main.py** for training a model using various strategies, including, no-defense, single mechanisms, pair-wise mechanisms and unified defense strategy.

### For Federated Learning Setting: 
- run **main.py** for training a model using various strategies, including, no-defense, single mechanisms, pair-wise mechanisms and unified defense strategy.

### Arguments: 
- **--model**: allows to select the architecture for the model
- **--dataset**: allows to select the dataset across CIFAR10, MNIST and Fashion-MNIST 
- **--dp**: incorporates differential privacy into the training algorithms
- **--dp_epsilon**: specifies the dp epsilon
- **--adv**: incorporates the adversarial training
- **--pgd_eps**: epsilon for PGD
- **--pgd_attack_steps**: number of attack steps for PGD
- **--pgd_step_size**: step size for PGD
- **--wm**: watermarks the model 
- Many other arguments used as part of the traditional ML training process, comprising of learning rate, number of epochs, batch size and etc.
- Some arguments were specific to FL setting, i.e., local epochs, global epochs, number of clients, and fraction of clients selected at every iteration for aggregation step. 
- Examples of centralised and FL setting training can be found in **cluster.sh** or **centralised/script.sh**