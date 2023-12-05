"""Run a stimulation with the predictive coding model on the CPU and plot the results.

This script reproduces Figures 5 and 6A of:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""

from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import PCModel  # CPU model
from weights_nour_eddine_2023 import get_weights

# Make sure to set this to where you've downloaded Samer's data package to.
data_path = "./helper_txt_files"

# You can play with these to run more or fewer simulation steps
n_pre_iterations = 2
n_iterations = 20

# Instantiate the model
weights = get_weights(data_path)
m = PCModel(weights, batch_size=512)

# Grab the list of words in the experiment. We will use only the first 512 as inputs.
with open(f"{data_path}/1579words_words.txt") as f:
    lex = f.read()
    lex = lex.strip().split("\n")
input_batch = lex[:512]  # only use the first 512 words to test the model

# Run through the standard 512 words once to pre-activate the model units.
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
m_standard = deepcopy(m)  # designate as the "standard" initialized model

##
# Experiment 1: Repetition priming
# We will feed the same words through the model again, which should elicit very small
# prediction errors, since the units have been correctly pre-activated.

m = deepcopy(m_standard)
pred_err_repeat = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_repeat.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
    pred_err_repeat.append(m.get_lex_sem_prederr())

# As a contrast to the repeated words, we will now feed through the words in a different
# order (non-repetition), which should elicit a much larger prediction error, since the
# units have now been incorrectly pre-activated.
np.random.shuffle(input_batch)

m = deepcopy(m_standard)
pred_err_non_repeat = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_non_repeat.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
    pred_err_non_repeat.append(m.get_lex_sem_prederr())

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
m = deepcopy(m_standard)
pred_err_sem_rel = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_sem_rel.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(sem_related)
    pred_err_sem_rel.append(m.get_lex_sem_prederr())

# Semantically unrelated
m = deepcopy(m_standard)
pred_err_sem_unrel = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_sem_unrel.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(sem_unrelated)
    pred_err_sem_unrel.append(m.get_lex_sem_prederr())

##
# Experiment 3: Cloze probability
# Preactive the units by setting the control units to a prediction of the upcoming word.
# How accurate this prediction is will be determined by the cloze probability of the
# word.

# Try various cloze probabilities
cloze_prob_experiments = dict(
    low=1 / len(m.lex_units),
    mid_low=0.25,
    mid_high=0.5,
    high=0.99,
)
pred_err_cloze = {name: [] for name in cloze_prob_experiments}

for name, cloze_prob in cloze_prob_experiments.items():
    # Start from a fresh model.
    m.reset()

    # Preactivate the model using the control (dummy) units.
    for n in tqdm(range(n_iterations), unit="step"):
        m(clamp_orth=None, clamp_ctx=input_batch, cloze_prob=cloze_prob)
        pred_err_cloze[name].append(m.get_lex_sem_prederr())

    # Run through the standard batch of words.
    for n in tqdm(range(n_iterations), unit="step"):
        m(input_batch)
        pred_err_cloze[name].append(m.get_lex_sem_prederr())

##
# Plot the result.
fig, axes = plt.subplots(ncols=3, figsize=(12, 3), sharex=True, sharey=True)
iters_to_plot = 22

axes[0].plot(pred_err_repeat[-iters_to_plot:], label="Repeated")
axes[0].plot(pred_err_non_repeat[-iters_to_plot:], label="Non-repeated")
axes[0].set_title("Repetition priming")
axes[0].set_xlabel("Number of Iterations")
axes[0].set_ylabel("Lexico-semantic PE")
axes[0].legend()

axes[1].plot(pred_err_sem_rel[-iters_to_plot:], label="Related")
axes[1].plot(pred_err_sem_unrel[-iters_to_plot:], label="Unrelated")
axes[1].set_title("Semantic priming")
axes[1].set_xlabel("Number of Iterations")
axes[1].set_ylabel("Lexico-semantic PE")
axes[1].legend()

for name, pred_err in pred_err_cloze.items():
    axes[2].plot(pred_err[-iters_to_plot:], label=name)
axes[2].set_title("Cloze probability")
axes[2].set_xlabel("Number of Iterations")
axes[2].set_ylabel("Lexico-semantic PE")
axes[2].legend()

fig.tight_layout()
