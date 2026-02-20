import torch
import os

if torch.cuda.is_available():
    cuda_device = torch.device("cuda")
    x = torch.ones(1, device=cuda_device)
    print (x)
else:
    print ("NVIDIA CUDA device not found.")