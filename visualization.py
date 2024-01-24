import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import trange

from weights_nour_eddine_2023 import get_weights, get_orth_repr

# Make sure to set this to where you've downloaded Samer's data package to.
data_path = "./helper_txt_files"

weights = get_weights(data_path)

batch_size = 512
#n_sem = 300

# Grab the list of words in the experiment. We will use only the first 512 as inputs.
with open(f"{data_path}/1579words_words.txt") as f:
    lex = f.read()
    lex = lex.strip().split("\n")
input_batch = lex[:batch_size]  # only use the first 512 words to test the model

orth_state = np.array([get_orth_repr(word) for word in input_batch])
orth_pred = np.zeros_like(orth_state)
orth_err = orth_state - orth_pred

lex_state = np.zeros((batch_size, len(weights.lex_units)))
lex_pred = np.zeros((batch_size, len(weights.lex_units)))
lex_err = np.zeros((batch_size, len(weights.lex_units)))

sem_state = np.zeros((batch_size, len(weights.sem_units)))
sem_err = np.zeros((batch_size, len(weights.sem_units)))
#weights.V_sem_lex = np.random.rand(len(weights.lex_units), n_sem)
#weights.V_sem_lex /= weights.V_sem_lex.sum(axis=1, keepdims=True)

prederr = [np.linalg.norm(lex_err, axis=1).sum() + np.linalg.norm(sem_err, axis=1).sum()]

##
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(19, 11))
ax_orth_state = axes[0, 0]
ax_orth_pred = axes[1, 0]
ax_orth_err = axes[2, 0]
ax_lex_state = axes[0, 1]
ax_lex_pred = axes[1, 1]
ax_lex_err = axes[2, 1]
ax_sem_state = axes[0, 2]
ax_sem_err = axes[1, 2]
ax_prederr = axes[2, 2]

ax_orth_state.set_title("Orth state")
pcm_orth_state = ax_orth_state.imshow(orth_state, aspect="auto", interpolation="nearest")
cb_orth_state = fig.colorbar(pcm_orth_state, ax=ax_orth_state)

ax_orth_pred.set_title("Orth pred")
pcm_orth_pred = ax_orth_pred.imshow(orth_pred, aspect="auto", interpolation="nearest")
cb_orth_pred = fig.colorbar(pcm_orth_pred, ax=ax_orth_pred)

ax_orth_err.set_title("Orth err")
pcm_orth_err = ax_orth_err.imshow(orth_err, aspect="auto", interpolation="nearest")
cb_orth_err = fig.colorbar(pcm_orth_err, ax=ax_orth_err)

ax_lex_state.set_title("lex state")
pcm_lex_state = ax_lex_state.imshow(lex_state, aspect="auto", interpolation="nearest")
cb_lex_state = fig.colorbar(pcm_lex_state, ax=ax_lex_state)

ax_lex_pred.set_title("lex pred")
pcm_lex_pred = ax_lex_pred.imshow(lex_pred, aspect="auto", interpolation="nearest")
cb_lex_pred = fig.colorbar(pcm_lex_pred, ax=ax_lex_pred)

ax_lex_err.set_title("lex td err")
pcm_lex_err = ax_lex_err.imshow(lex_err, aspect="auto", interpolation="nearest")
cb_lex_err = fig.colorbar(pcm_lex_err, ax=ax_lex_err)

ax_sem_state.set_title("sem state")
pcm_sem_state = ax_sem_state.imshow(sem_state, aspect="auto", interpolation="nearest")
cb_sem_state = fig.colorbar(pcm_sem_state, ax=ax_sem_state)

ax_sem_err.set_title("sem td err")
pcm_sem_err = ax_sem_err.imshow(sem_err, aspect="auto", interpolation="nearest")
cb_sem_err = fig.colorbar(pcm_sem_err, ax=ax_sem_err)

ax_prederr.set_title("pred err")

fig.tight_layout()

##

lex_state += 0.001 * (orth_err @ weights.V_lex_orth - lex_err)
sem_state += 0.001 * lex_err @ weights.V_sem_lex
sem_err = lex_err @ weights.V_sem_lex

orth_pred = lex_state @ weights.V_lex_orth.T
orth_err = orth_state - orth_pred

lex_pred = sem_state @ weights.V_sem_lex.T
lex_err = lex_state - lex_pred

pcm_orth_pred.set_data(orth_pred)
pcm_orth_pred.set_clim(orth_pred.min(), orth_pred.max())
cb_orth_pred.set_ticks(np.linspace(orth_pred.min(), orth_pred.max(), 5))

pcm_orth_err.set_data(orth_err)
pcm_orth_err.set_clim(orth_err.min(), orth_err.max())
cb_orth_err.set_ticks(np.linspace(orth_err.min(), orth_err.max(), 5))

pcm_lex_state.set_data(lex_state)
pcm_lex_state.set_clim(lex_state.min(), lex_state.max())
cb_lex_state.set_ticks(np.linspace(lex_state.min(), lex_state.max(), 5))

pcm_lex_pred.set_data(lex_pred)
pcm_lex_pred.set_clim(lex_pred.min(), lex_pred.max())
cb_lex_pred.set_ticks(np.linspace(lex_pred.min(), lex_pred.max(), 5))

pcm_lex_err.set_data(lex_err)
pcm_lex_err.set_clim(lex_err.min(), lex_err.max())
cb_lex_err.set_ticks(np.linspace(lex_err.min(), lex_err.max(), 5))

pcm_sem_state.set_data(sem_state)
pcm_sem_state.set_clim(sem_state.min(), sem_state.max())
cb_sem_state.set_ticks(np.linspace(sem_state.min(), sem_state.max(), 5))

pcm_sem_err.set_data(sem_err)
pcm_sem_err.set_clim(sem_err.min(), sem_err.max())
cb_sem_err.set_ticks(np.linspace(sem_err.min(), sem_err.max(), 5))

acc = np.mean(np.argmax(lex_state, axis=1) == np.arange(batch_size))
print(f"Accuracy: {acc}")

prederr.append(np.linalg.norm(lex_err, axis=1).sum() + np.linalg.norm(sem_err, axis=1).sum())
ax_prederr.clear()
ax_prederr.plot(prederr)
##

for _ in trange(100):
    lex_state += 0.001 * (orth_err @ weights.V_lex_orth - lex_err)
    sem_state += 0.001 * lex_err @ weights.V_sem_lex
    sem_err = lex_err @ weights.V_sem_lex

    orth_pred = lex_state @ weights.V_lex_orth.T
    orth_err = orth_state - orth_pred

    lex_pred = sem_state @ weights.V_sem_lex.T
    lex_err = lex_state - lex_pred

    prederr.append(np.linalg.norm(lex_err, axis=1).sum() + np.linalg.norm(sem_err, axis=1).sum())
