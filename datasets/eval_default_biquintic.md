# Steps to recreate both eval_default_biquintic datasets

created on version 1.2.3


```python
random_seed = 1
train_domains = [-23, 0, 23, 45, 90, 180]
test_domains = [-135, -90, -45, 10, 30, 60, 75, 135]

all_domains = sorted(set(train_domains) | set(test_domains))   # all unique angles, sorted (same as in 'eval_default_v1')
train_set = RotatedMNIST(domains=train_domains, 
                            train=True, 
                            seed=random_seed, 
                            val_set_size=1000, 
                            normalize=True, 
                            add_noise=False, 
                            interpolation='biquintic')
test_set = RotatedMNIST(domains=all_domains, 
                        train=False, 
                        seed=random_seed, 
                        normalize=True, 
                        add_noise=False, 
                        interpolation='biquintic')
```
