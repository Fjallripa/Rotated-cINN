# Steps to recreate the train_default_biquintic dataset

created on version 1.2.3


```python
random_seed = 1
train_domains = [-23, 0, 23, 45, 90, 180]

train_set = RotatedMNIST(domains=train_domains, 
                            train=True, 
                            seed=random_seed, 
                            val_set_size=1000, 
                            normalize=True, 
                            add_noise=True,
                            transform = AddGaussianNoise(0, 0.08),
                            interpolation='biquintic')
```
