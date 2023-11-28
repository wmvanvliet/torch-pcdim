"""Run a stimulation with the predictive coding model on the GPU and plot the results.

This script reproduces Figure 5A of:

Eddine, Samer Nour, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch_model import PCModel  # GPU model
from weights_nour_eddine_2023 import get_weights

# Make sure to set this to where you've downloaded Samual Eddine Nour's data package to.
data_path = "./helper_txt_files"

# You can play with these to run more or fewer simulation steps
n_pre_iterations = 2
n_iterations = 20

# Instantiate the model
weights = get_weights(data_path)
m = PCModel(weights, batch_size=512).cuda()

# Grab the list of words in the experiment. We will use only the first 512 as inputs.
with open(f"{data_path}/1579words_words.txt") as f:
    lex = f.read()
    lex = lex.strip().split("\n")
input_batch = lex[:512]  # only use the first 512 words to test the model

# Run through the standard 512 words once to pre-activate the model units.
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

preinit_state = m.state_dict()  # designate this as the "standard" initialized model

##
# Experiment 1: Repetition priming
# We will feed the same words through the model again, which should elicit very small
# prediction errors, since the units have been correctly pre-activated.

m.load_state_dict(preinit_state)
pred_err_repeat = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m(None)
    pred_err_repeat.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
    pred_err_repeat.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

# As a contrast to the repeated words, we will now feed through the words in a different
# order (non-repetition), which should elicit a much larger prediction error, since the
# units have now been incorrectly pre-activated.
np.random.shuffle(input_batch)

m.load_state_dict(preinit_state)
pred_err_non_repeat = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m(None)
    pred_err_non_repeat.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
    pred_err_non_repeat.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

##
# Experiment 2: Semantic priming
# Contrast words that are semantically related to the words used to pre-activate the
# units, versus words that are semantically unrelated. Semantic relatedness is judged by
# the number of shared semantic features (8 shared features = semantically related).
sem_features = (m.weights.W_lex_sem > 0).astype("float")
n_shared_feats = sem_features.T @ sem_features
sem_related_ind = np.argwhere(n_shared_feats == 8)[:, 1]
sem_unrelated_ind = np.loadtxt(
    f"{data_path}/semunrelated_matlab.csv", skiprows=1, dtype="int"
)
sem_related = [lex[i] for i in sem_related_ind]
sem_unrelated = [lex[i] for i in sem_unrelated_ind]

# Semantically related
m.load_state_dict(preinit_state)
pred_err_sem_rel = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m(None)
    pred_err_sem_rel.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(sem_related)
    pred_err_sem_rel.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

# Semantically unrelated
m.load_state_dict(preinit_state)
pred_err_sem_unrel = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m(None)
    pred_err_sem_unrel.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(sem_unrelated)
    pred_err_sem_unrel.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

##
# Plot the result.
fig, axes = plt.subplots(ncols=2, figsize=(8, 3))

axes[0].plot(pred_err_repeat, label="Repeated")
axes[0].plot(pred_err_non_repeat, label="Non-repeated")
axes[0].set_title("Repetition priming")
axes[0].set_xlabel("Number of Iterations")
axes[0].set_ylabel("Lexico-semantic PE")
axes[0].legend()

axes[1].plot(pred_err_sem_rel, label="Related")
axes[1].plot(pred_err_sem_unrel, label="Unrelated")
axes[1].set_title("Semantic priming")
axes[1].set_xlabel("Number of Iterations")
axes[1].set_ylabel("Lexico-semantic PE")
axes[1].legend()

fig.tight_layout()
