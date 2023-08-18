# Steps to recreate both eval_30_deg_spacing datasets

created on version 1.2.5

```python
random_seed = 1
train_domains = [-60, -30, 0, 30, 60]   # 30° spacing of 5 domains
test_domains = [15, 45, 75, 90, 105, 120, 150, 180]

all_domains = sorted(set(train_domains) | set(test_domains))   # all unique angles, sorted
train_set = RotatedMNIST(domains=train_domains, train=True, seed=random_seed, val_set_size=1000, normalize=True, add_noise=False)
test_set = RotatedMNIST(domains=all_domains, train=False, seed=random_seed, normalize=True, add_noise=False)
```
