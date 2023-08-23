# Steps to recreate both eval_unrotated datasets

created on version 1.2

```python
random_seed = 1
train_domains = [0]
test_domains = [0]

all_domains = sorted(set(train_domains) | set(test_domains))   # all unique angles, sorted
train_set = RotatedMNIST(domains=train_domains, train=True, seed=random_seed, val_set_size=1000, normalize=True, add_noise=False)
test_set = RotatedMNIST(domains=all_domains, train=False, seed=random_seed, normalize=True, add_noise=False)
```
