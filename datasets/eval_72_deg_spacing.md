# Steps to recreate both eval_72_deg_spacing datasets

created on version 1.2.5

```python
random_seed = 1
train_domains = [-144, -72, 0, 72, 144]   # 72° spacing of 5 domains
test_domains = [-108, -54, -36, -18, 18, 36, 54, 108, 180]

all_domains = sorted(set(train_domains) | set(test_domains))   # all unique angles, sorted
train_set = RotatedMNIST(domains=train_domains, train=True, seed=random_seed, val_set_size=1000, normalize=True, add_noise=False)
test_set = RotatedMNIST(domains=all_domains, train=False, seed=random_seed, normalize=True, add_noise=False)
```
