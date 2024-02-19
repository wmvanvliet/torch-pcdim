import torch
from torch_predcoding import InputLayer, OutputLayer
from scipy.io import loadmat
from torchvision import datasets, transforms


# m = loadmat("../ImageRecognition/classify_images_dim_MNIST_n5000.mat")
# W, V, X, Y = [torch.tensor(m[x]).float() for x in ["W", "V", "X", "Y"]]
#
# l1 = InputLayer(n_units=len(X))
# l2 = OutputLayer(n_in=len(X), n_units=len(Y), bu_weights=W.T, td_weights=V)

transform = transforms.ToTensor()
dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)

l1 = InputLayer(n_units=28 * 28).cuda()
l2 = OutputLayer(n_in=28 * 28, n_units=5000).cuda()


def measure_nse(observed, simulated):
    return 1 - torch.sum((observed - simulated) ** 2) / torch.sum(
        (observed - simulated.mean()) ** 2
    )


def measure_sparsity_hoyer(x):
    sqrt_len = torch.sqrt(torch.as_tensor(x.shape[1]))
    return (sqrt_len - torch.norm(x, 1, dim=1) / torch.norm(x, 2, dim=1)) / (
        sqrt_len - 1
    )


cycs = 2_400_000
show = 100

max_y = torch.as_tensor(0.0).cuda()
sum_nse = torch.as_tensor(0.0).cuda()
sum_sparsity = torch.as_tensor(0.0).cuda()

for cyc in range(cycs):
    X, _ = dataset1[torch.randint(0, len(dataset1), (1,))[0]]
    X = X.view(1, -1).cuda()
    l1.clamp(X)
    l2.reset()

    for t in range(1, 51):
        r = l2.backward()
        l1.backward(r)
        e = l1.forward(X)
        l2.forward(e)
        y = l2.state

    l2.train_weights(e, lr=0.1)

    max_y = torch.maximum(max_y, l2.state.max())
    sum_nse += measure_nse(X, l2.backward())
    sum_sparsity += measure_sparsity_hoyer(l2.state).mean()

    if (cyc + 1) % show == 0 or cyc == cycs - 1:
        print(
            f"{cyc + 1:06d} ymax={max_y:.3f} NSE={sum_nse / show:.3f} sparsity={sum_sparsity / show:.3f}"
        )
        max_y = torch.as_tensor(0.0).cuda()
        sum_nse = torch.as_tensor(0.0).cuda()
        sum_sparsity = torch.as_tensor(0.0).cuda()
