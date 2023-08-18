# Settings to recreate the training of 'recreation_default_spacing'

created on version 1.2.5

Description: 
It's the same model as 'recreation_of_og_cinn' but it's trained on the 'train_default_biquintic' dataset in order to see the effects of domain generalization. 

```python
# General settings
random_seed_ = 1

# Model parameters
model_name = "recreation_default_spacing"
model_subnet = "original"

# Dataset parameters
dataset_name = "train_default_biquintic"
## for details see 'datasets/train_default_biquintic.md'

# Training parameters
epochs = 60
batch_size = 256
optimizer = Adam(
    learning_rate = 5e-4, 
    weight_decay = 1e-5)
scheduler = MultiStepLR(
    milestones = [20, 40],
    gamma = 0.1)
loss_function = loss.neg_loglikelihood
```