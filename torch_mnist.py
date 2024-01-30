import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from torch_predcoding import InputLayer, MiddleLayer, OutputLayer


def train(args, model, device, train_loader, epoch, n_iter=20):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.reset(batch_size=len(data))
        model.clamp(data, target)
        for i in range(n_iter):
            output = model(data)
            model.backward()
            if i % 5 == 0:
                model.train_weights(data)
        model.release_clamp()
        model.reset(batch_size=len(data))
        for i in range(n_iter):
            output = model(data)
            model.backward()
        train_loss = F.nll_loss(
            output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).float().mean().item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    train_loss,
                    100.0 * correct,
                )
            )
            if batch_idx == 20:
                break

    model.release_clamp()


def test(model, device, test_loader, n_iter=20):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.release_clamp()
            model.reset(batch_size=len(data))
            for _ in range(n_iter):
                output = model(data)
                model.backward()
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
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
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
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
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {"batch_size": args.batch_size}
test_kwargs = {"batch_size": args.test_batch_size}
if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
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


class PCModel(nn.Module):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("input", InputLayer(batch_size=batch_size, n_units=28 * 28)),
                    (
                        "hidden",
                        MiddleLayer(batch_size=batch_size, n_units=500, n_in=28 * 28),
                    ),
                    (
                        "output",
                        OutputLayer(batch_size=batch_size, n_in=500, n_units=10),
                    ),
                ]
            )
        )

    def clamp(self, data, target):
        data = data.reshape(-1, 28 * 28)
        target = F.one_hot(target, 10).float()
        self.layers.input.clamp(data)
        self.layers.output.clamp(target)

    def release_clamp(self):
        self.layers.input.release_clamp()
        self.layers.output.release_clamp()

    def forward(self, x):
        bu_err = self.layers.input(x.reshape(-1, 28 * 28))
        bu_err = self.layers.hidden(bu_err)
        return self.layers.output(bu_err)

    def backward(self):
        rec = self.layers.output.backward()
        rec = self.layers.hidden.backward(rec)
        self.layers.input.backward(rec)

    def train_weights(self, x):
        bu_err_hidden = self.layers.input(x.reshape(-1, 28 * 28))
        bu_err_output = self.layers.hidden(bu_err_hidden)
        self.layers.hidden.train_weights(bu_err_hidden)
        self.layers.output.train_weights(bu_err_output)

    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        self.layers.input.reset(batch_size)
        self.layers.hidden.reset(batch_size)
        self.layers.output.reset(batch_size)


model = PCModel().to(device)
for epoch in range(1):
    train(args, model, device, train_loader, epoch, n_iter=20)
    test(model, device, test_loader, n_iter=20)
