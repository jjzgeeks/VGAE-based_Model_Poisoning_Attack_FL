# VGAE-based_Model_Poisoning_Attack_FL
These codes are about variational  graph autoencoder-based model poisoning attack against federated learning.

## Requirements
- Install requirements via  `pip install -r requirements.txt`


## How to run :point_down:
```
python FL_VGAE_Attack_main.py 
```
Note that the variable **num_clients_index** in the *FL_VGAE_Attack_main.py* must be kept consistent with the variable **input_dim** in *args.py*


## References
1. https://github.com/aswarth123/Federated_Learning_MNIST
2. https://github.com/DaehanKim/vgae_pytorch
3. https://arxiv.org/abs/1611.07308
