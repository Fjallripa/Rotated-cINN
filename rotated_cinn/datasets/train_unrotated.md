# Steps to recreate the train_unrotated dataset

created on version 1.2.5


```python
random_seed = 1
train_domains = [0]

train_set = RotatedMNIST(domains=train_domains, 
                            train=True, 
                            seed=random_seed, 
                            val_set_size=1000, 
                            normalize=True, 
                            add_noise=True,
                            transform = AddGaussianNoise(0, 0.08),
                            interpolation='biquintic')
```
