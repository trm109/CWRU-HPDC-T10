# Detects the presence of a CUDA device.
import torch.cuda as cuda

if cuda.is_available():
    device_ct = cuda.device_count()

    # Devices are indexed by zero. Print device details:
    print(f'{device_ct} devices detected.')

    for idx in range(device_ct):
        gpu = cuda.device(idx)
        print(f'GPU device: {gpu}\n{cuda.get_device_properties(gpu)}')
else:
    print('No GPU detected!')
