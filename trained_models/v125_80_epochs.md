# Settings to recreate the training of 'v125_80_epochs'

created on version 1.2.5

Description: 
This is the best performing model that I have been able to create so far (v1.2.5).
Compared to 'recreation_og_cinn', it uses
* a subnet of double the with and double the depth, 
* 80 instead of 60 training epochs, and
* a cosine annealing scheduler instead of the multistep one. 

```python
# General settings
random_seed_ = 1

# Model parameters
model_name = "v125_80_epochs"
model_subnet = "og_two-deepter"

# Dataset parameters
dataset_name = "train_default_biquintic"
## for details see 'datasets/train_default_biquintic.md'

# Training parameters
epochs = 80
batch_size = 256
optimizer = Adam(
    learning_rate = 5e-4,   # initial learning rate
    weight_decay = 1e-5)
scheduler = CosineAnnealingLR(
    T_max = 80,   # = epochs
    eta_min = 5e-6)   # final learning rate
loss_function = loss.neg_loglikelihood
```