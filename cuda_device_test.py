# Detects the presence of a CUDA device!
import torch

if torch.cuda.is_available():
    gpu = torch.device('cuda:0')
    print(gpu)
else:
    print('No GPU detected!')