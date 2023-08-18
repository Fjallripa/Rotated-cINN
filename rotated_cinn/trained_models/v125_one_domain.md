# Settings to recreate the training of 'v125_one_domain'

created on version 1.2.5

Description: 
This is the same model as 'v125_80_epochs', but it's only on the 0° (unrotated) domain. It can be directly compared to 'recreation_og_cinn'.

```python
# General settings
random_seed_ = 1

# Model parameters
model_name = "v125_80_epochs"
model_subnet = "og_two-deepter"

# Dataset parameters
dataset_name = "train_unrotated"
## for details see 'datasets/train_unrotated.md'

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