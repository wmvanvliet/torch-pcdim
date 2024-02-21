"""Run a stimulation with the predictive coding model on the GPU and plot the results.

This script reproduces Figures 5 and 6A of:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime

from torch_model import N400Model  # GPU model
from weights_nour_eddine_2023 import get_weights

# Make sure to set this to where you've downloaded Samer's data package to.
data_path = "./helper_txt_files"

# You can play with these to run more or fewer simulation steps
n_pre_iterations = 2
n_iterations = 20
training_interval = 3
training_n_iterations = 100
batch_size = 512  # set how many inputs to test on.
training_batch_size = 1579  # set how many inputs to train on. I initially used 10 for debugging purposes

weights = get_weights(data_path)

# These weights are initialized randomly and will be learned through training
weights.W_orth_lex = torch.rand(*weights.W_orth_lex.shape) * 0.1
weights.V_lex_orth = weights.W_orth_lex.T.clone()

# Instantiate the model
m = N400Model(weights, batch_size=training_batch_size).cuda()

# Grab the list of words in the experiment. We will use only the first 512 as inputs,
# but all of them for training.
with open(f"{data_path}/1579words_words.txt") as f:
    lex = f.read()
    lex = lex.strip().split("\n")
training_batch = lex[:training_batch_size]
input_batch = lex[:batch_size]

'''
Preactivate the model using the control (dummy) units,
so that the model is biased to use particular lexical units to reconstruct each orthographic input
e.g. this enables the R-O-B-E orthographic units to activate the third lexical state unit, which is already linked to <robe> semantic features.
Without this constraint, R-O-B-E will learn to activate a random lexical unit (or a small set of them) that is very unlikely to be linked to <robe> features.
Conceptually, this is analogous to learning "what's the word for X?", where the semantics of X are already known (e.g. a cup), and it is known that there is some word for it.
'''
for n in tqdm(range(n_iterations), unit="step"):
    m(clamp_orth=None, clamp_ctx=training_batch, cloze_prob=0.99)

print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

# Train the model on all inputs
for n in tqdm(range(training_n_iterations), unit="step"):
    # clamping both ctx and orth during training further biases the model to activate particular lexical units in response to particular orthographic inputs, as described above
    m(clamp_orth=training_batch, clamp_ctx=training_batch, train_weights=n % training_interval == 0)
m.reset(batch_size)  # update batch size
trained_model = m.state_dict()
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

# Save the trained model
current_datetime = datetime.now().strftime("%m%d%y")
torch.save(trained_model, f'./helper_txt_files/trained_V_{current_datetime}.pt')

# Run through the input_batch to get a new pre-activated model, this time a trained one.
for n in tqdm(range(n_iterations), unit="step"):
    out = m(input_batch)
preact_state = m.state_dict()  # designate this as the "standard" initialized model

# Check whether the model outputs the correct words given the input
accuracy = np.mean(out.cpu().numpy().argmax(axis=1) == np.arange(batch_size))
print("Accuracy of the model after training:", accuracy)

##
# Experiment 1: Repetition priming
# We will feed the same words through the model again, which should elicit very small
# prediction errors, since the units have been correctly pre-activated.

m.load_state_dict(preact_state)
pred_err_repeat = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_repeat.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(input_batch)
    pred_err_repeat.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

# As a contrast to the repeated words, we will now feed through the words in a different
# order (non-repetition), which should elicit a much larger prediction error, since the
# units have now been incorrectly pre-activated.
np.random.shuffle(input_batch)

m.load_state_dict(preact_state)
pred_err_non_repeat = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
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
m.load_state_dict(preact_state)
pred_err_sem_rel = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_sem_rel.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(sem_related)
    pred_err_sem_rel.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

# Semantically unrelated
m.load_state_dict(preact_state)
pred_err_sem_unrel = []
for n in tqdm(range(n_pre_iterations), unit="step"):
    m("zeros")
    pred_err_sem_unrel.append(m.get_lex_sem_prederr())
for n in tqdm(range(n_iterations), unit="step"):
    m(sem_unrelated)
    pred_err_sem_unrel.append(m.get_lex_sem_prederr())
print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

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
    print("GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")

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
