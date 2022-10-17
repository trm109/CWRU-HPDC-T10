# Torchvision test - loading a CIFAR10 dataset.
# =============================================
# CSDS 438, Group 10
# Josh, Theo, Max, Priyan, David, Kane

import torch.utils.data.dataloader as dloader
import torchvision.datasets.cifar as cifar
import torchvision.transforms as T

# Step 1 - We normalize RGB values to have a mean and standard deviation of 0.5.
#          We then create a training set out of torchvision's built-in CIFAR10 dataset,
#          which contains a bunch of 32x32 images containing the following classes:
#
#          airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
#
#          More details (and example images) can be found here:
#          https://www.cs.toronto.edu/~kriz/cifar.html
transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = cifar.CIFAR10(root='./data', download=True, transform=transform)
train_loader = dloader.DataLoader(train_set, batch_size=4, num_workers=2)
