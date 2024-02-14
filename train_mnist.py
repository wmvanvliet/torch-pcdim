import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm


def train(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), unit="batches"
    ):
        data = data.to(device)
        target = F.one_hot(target, 10).float().to(device)
        optimizer.zero_grad()
        pred = model(target)
        loss = criterion(pred, data)
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            p.data.clamp_(0)
        # print(f"Batch loss: {loss:.4f}\n")


def test(model, device, test_loader, n_iter=20):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data = data.to(device)
        target = F.one_hot(target, 10).float().to(device)
        pred = model(target)
        test_loss += criterion(pred, data).item()
    test_loss /= len(test_loader)
    print(f"Average loss: {test_loss:.4f}\n")


# Training settings
parser = argparse.ArgumentParser(description="PyTorch predictive coding MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for training (default: 512)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    metavar="N",
    help="number of epochs to train (default: 30)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1,
    metavar="LR",
    help="initial learning rate (default: 1)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
)
parser.add_argument(
    "--step-down",
    type=int,
    default=10,
    metavar="LR",
    help="step down learning rate after this amount of epochs (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
args = parser.parse_args()

# Torch configuration
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# Load the dataset
train_kwargs = {"batch_size": args.batch_size}
test_kwargs = {"batch_size": args.test_batch_size}
if use_cuda:
    cuda_kwargs = {"num_workers": 0, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("./data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

# Build and train the model
model = nn.Sequential(
    OrderedDict(
        [
            ("output", nn.Linear(10, 16 * 14 * 14, bias=False)),
            ("hidden2", nn.Unflatten(1, (16, 14, 14))),
            ("hidden1", nn.Upsample((28, 28))),
            (
                "hidden0",
                nn.ConvTranspose2d(16, 1, bias=False, kernel_size=5, padding=2),
            ),
        ]
    )
).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

# Save trained model
checkpoint = dict(args.__dict__)
checkpoint["state_dict"] = model.state_dict()
torch.save(checkpoint, "data/MNIST/trained_forward_model.pkl")
