import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator,initialize_weights

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNEL_IMG = 1
Z_DIM = 100
NUM_EPOCHS=5
FEATURES_DISC, FEATHURE_DEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [
                0.5 for _ in range(CHANNEL_IMG)
            ],
            [
                0.5 for _ in range(CHANNEL_IMG)
            ]
        ),
    ]
)

datasets = datasets.MNIST(
    'dataset/',
    train=True,
    transform=transforms,
    download=True
)