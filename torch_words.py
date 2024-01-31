import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from torch_predcoding import PCModel
from dataloaders import WebDataset


def train(args, model, device, train_loader, epoch, n_iter=20, freq=5, lr=0.01):
    model.train()
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), unit="batches"
    ):
        data = data.view(-1, 224 * 224).to(device)
        target = F.one_hot(target, 1000).float().to(device)
        model.reset(batch_size=len(data))
        model.clamp(data, target)
        for i in range(n_iter):
            model(data)
            model.backward()
            if i % freq == 0:
                model.train_weights(data, lr=lr)
    model.release_clamp()


def test(model, device, test_loader, n_iter=20):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 224 * 224).to(device)
            target = target.to(device)
            model.release_clamp()
            model.reset(batch_size=len(data))
            for _ in range(n_iter):
                output = model(data.view(-1, 224 * 224))
                model.backward()
            test_loss += F.nll_loss(
                F.log_softmax(output, dim=1), target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set performance:"
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


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
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--step-down",
    type=int,
    default=10,
    metavar="LR",
    help="step down learning rate after this amount of epochs (default: 10)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
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
    cuda_kwargs = {"num_workers": 0, "pin_memory": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
dataset1 = WebDataset("/l/vanvlm1/datasets/1kwords-freq-no-noise", train=True, shuffle=True, transform=transform)
dataset2 = WebDataset("/l/vanvlm1/datasets/1kwords-freq-no-noise", train=False, shuffle=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

# Build and train the model
model = PCModel(n_units=[224 * 224, 4096, 4096, 10000], batch_size=args.batch_size).to(device)
lr = args.lr
for epoch in range(args.epochs):
    if epoch % args.step_down == 0:
        lr /= 10
    train(args, model, device, train_loader, epoch, n_iter=100, freq=10, lr=lr)
    test(model, device, test_loader, n_iter=20)

# Save trained model
checkpoint = dict(args.__dict__)
checkpoint["state_dict"] = model.state_dict()
torch.save(checkpoint, "trained_word_model.pkl")
