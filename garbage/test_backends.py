import torch
import numpy as np

a = torch.tensor([1, 2, 3, 4, 5])
b = np.array([0, 4, 3, 2, 1])

print(list(np.argsort(b)))
print(sorted(range(len(b)), key=b.__getitem__, reverse=False))
