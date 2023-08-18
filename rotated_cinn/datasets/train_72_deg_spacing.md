# Steps to recreate the train_72_deg_spacing dataset

created on version 1.2.5


```python
random_seed = 1
train_domains = [-144, -72, 0, 72, 144]   # 72° spacing of 5 domains

train_set = RotatedMNIST(domains=train_domains, 
                            train=True, 
                            seed=random_seed, 
                            val_set_size=1000, 
                            normalize=True, 
                            add_noise=True,
                            transform = AddGaussianNoise(0, 0.08),
                            interpolation='biquintic')
```
