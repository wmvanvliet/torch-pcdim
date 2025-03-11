import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from torch_pcdim.models import PCModel
from torch_pcdim.layers import InputLayer, MiddleLayer, OutputLayer


def train(args, model, device, train_loader, epoch, n_iter=20, freq=5, lr=0.01):
    model.train()
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), unit="batches"
    ):
        data = data.view(-1, 28 * 28).to(device)
        target = F.one_hot(target, 10).float().to(device)
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
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            n += len(data)
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            model.release_clamp()
            model.reset(batch_size=len(data))
            for _ in range(n_iter):
                output = model(data.view(-1, 28 * 28))
                model.backward()
            test_loss += F.nll_loss(
                F.log_softmax(output, dim=1), target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n

    print(
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
            test_loss,
            correct,
            n,
            100.0 * correct / n,
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
    default=10,
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
    default=15,
    metavar="LR",
    help="step down learning rate after this amount of epochs (default: 15)",
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
    cuda_kwargs = {"num_workers": 0, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.ToTensor()
dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("./data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

# Build and train the model
model = PCModel(
    [
        InputLayer(n_units=28 * 28, batch_size=args.batch_size),
        MiddleLayer(n_in=28 * 28, n_units=1024, batch_size=args.batch_size),
        OutputLayer(n_in=1024, n_units=10, batch_size=args.batch_size),
    ]
).to(device)

lr = args.lr
for epoch in range(args.epochs):
    if epoch % args.step_down == 0:
        lr /= 10
    train(args, model, device, train_loader, epoch, n_iter=200, freq=50, lr=lr)
    test(model, device, test_loader, n_iter=100)

# Save trained model
checkpoint = dict(args.__dict__)
checkpoint["state_dict"] = model.state_dict()
torch.save(checkpoint, "data/MNIST/trained_model.pkl")

# Run a batch through the model and plot prediction errors at each layer
model.reset(batch_size=128)
model.release_clamp()
data, target = next(iter(train_loader))
data, target = data[:128], target[:128]
data, target = data.to(device), target.to(device)
errs = []
for i in range(50):
    with torch.no_grad():
        output = model(None)
        model.backward()
for i in range(100):
    with torch.no_grad():
        model.backward()
        output = model(data.view(-1, 28 * 28))
        e = []
        for l in model.layers:
            if hasattr(l, "bu_err"):
                e.append((l.bu_err).mean().detach().item())
            elif hasattr(l, "pred_err"):
                e.append((l.pred_err).mean().detach().item())
        errs.append(e)
errs = np.array(errs).T
fig, axes = plt.subplots(ncols=len(errs), figsize=(10, 4))
for ax, e in zip(axes, errs):
    ax.plot(e)
plt.savefig("prederr.png")

# Plot reconstruction of the input
grid_rec = make_grid(
    model.layers[0].reconstruction.detach().cpu().view((-1, 1, 28, 28))
)
grid_data = make_grid(data.cpu())
fig, axes = plt.subplots(ncols=2, figsize=(8, 8))
axes[0].imshow(grid_data.permute(1, 2, 0))
axes[0].set_title("input")
axes[0].set_axis_off()
axes[1].imshow(grid_rec.permute(1, 2, 0))
axes[1].set_title("reconstruction")
axes[1].set_axis_off()
plt.savefig("reconstruction.png")

# Plot "dream" reconstruction of the digits by clamping the numbers 0-9 on the outputs
# and letting the model converge.
model.reset(batch_size=10)
model.release_clamp()
target = (
    F.one_hot(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 10).float().to(device) * 9
)
model.clamp(input_data=None, output_data=target)
recons = []
for i in range(20):
    output = model(None)
    model.backward()
    recons.append(model.layers.input.state.cpu().reshape(10, 28, 28))
fig, axes = plt.subplots(nrows=10, ncols=20, figsize=(20, 10))
for i, r in enumerate(recons):
    for j in range(10):
        axes[j][i].imshow(r[j], vmin=0, vmax=1)
        axes[j][i].axis("off")
    axes[0][i].set_title(f"{i:02d}")
plt.tight_layout()
plt.savefig("mnist_reconstructions.png")
