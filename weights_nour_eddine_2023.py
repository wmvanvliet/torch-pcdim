"""Functions to obtain the weights for the predictive coding model of the N400.

These weights reproduce:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""

from dataclasses import dataclass
from itertools import product
from string import ascii_lowercase

import numpy as np


@dataclass
class Weights:
    """Dataclass containing the weights of the Nour Eddine et al. (2023) model."""

    orth_units: list[str]
    lex_units: list[str]
    sem_units: list[str]
    ctx_units: list[str]
    W_orth_lex: np.ndarray
    W_lex_sem: np.ndarray
    W_sem_ctx: np.ndarray
    V_lex_orth: np.ndarray
    V_sem_lex: np.ndarray
    V_ctx_sem: np.ndarray
    I_lex_sem: np.ndarray
    I_sem_ctx: np.ndarray
    I_orth_lex: np.ndarray
    freq: np.ndarray


def get_weights(data_path):
    """Obtain the weight matrices to reproduce Nour Eddine et al. (2023).

    Parameters
    ----------
    data_path : str
        The path to the ``helper_txt_files`` folder.

    Returns
    -------
    weights : Weights
        A dataclass with the following attributes:
          - ``orth_units`` string names for the orthographic units
          - ``lex_units`` string names for the lexical units
          - ``sem_units`` string names for the semantic units
          - ``ctx_units`` string names for the control (dummy) units
          - ``W_orth_lex`` (normalized) weights connecting orthographic to lexical units
          - ``W_lex_sem`` (normalized) weights connecting lexical to semantic units
          - ``W_sem_ctx`` (normalized) weights connecting semantic to control units
          - ``V_lex_orth`` backward weights connecting lexical to orthographic units
          - ``V_sem_lex`` backward weights connecting semantic to lexical units
          - ``V_ctx_sem`` backward weights connecting control to semantic units
          - ``I_lex_sem`` normalization matrix used for ``W_lex_sem``
          - ``I_sem_ctx`` normalization matrix used for ``W_sem_ctx``
          - ``I_orth_lex`` normalization matrix used for ``W_orth_lex``
          - ``freq`` frequency information for the lexical units
    """
    # orthographic units
    orth_units = [
        f"{letter}{pos}" for pos, letter in product([1, 2, 3, 4], ascii_lowercase)
    ]
    # lexical units
    with open(f"{data_path}/1579words_words.txt") as f:
        lex_units = f.read()
        lex_units = lex_units.strip().split("\n")
    # semantic units
    with open(f"{data_path}/semantic_features.txt") as f:
        sem_units = f.read()
        sem_units = sem_units.strip().split("\n")
    # control (dummy) units
    ctx_units = list(lex_units)

    # Assemble weight matrices
    sem_features = _assemble_semantic_features(data_path, lex_size=len(lex_units))
    W_orth_lex = np.array([get_orth_repr(word) for word in lex_units])
    W_lex_sem = sem_features
    W_sem_ctx = sem_features.T

    # Compute normalizers
    I_orth_lex = np.diag(1 / (np.sum(W_orth_lex, axis=1) + 1))
    I_lex_sem = np.diag(1 / (np.sum(W_lex_sem, axis=1) + 1))
    I_sem_ctx = np.diag(1 / np.sum(W_sem_ctx, axis=1))

    # Compute feedback weights
    V_ctx_sem = W_sem_ctx.T
    V_sem_lex = W_lex_sem.T
    V_lex_orth = W_orth_lex.T

    # this is wrong
    V_sem_lex /= V_sem_lex.sum(axis=1, keepdims=True)

    # Frequency information
    freq = np.loadtxt(f"{data_path}/1579words_freq_values.txt")

    return Weights(
        orth_units=orth_units,
        lex_units=lex_units,
        sem_units=sem_units,
        ctx_units=ctx_units,
        W_orth_lex=W_orth_lex,
        W_lex_sem=W_lex_sem,
        W_sem_ctx=W_sem_ctx,
        V_ctx_sem=V_ctx_sem,
        V_sem_lex=V_sem_lex,
        V_lex_orth=V_lex_orth,
        I_orth_lex=I_orth_lex,
        I_lex_sem=I_lex_sem,
        I_sem_ctx=I_sem_ctx,
        freq=freq,
    )


def get_orth_repr(word):
    """Get the orthographic representation of a word.

    In the Nour Eddine et al. (2023) model, the orthographic units consist of a letter
    bank of 4 groups of 26 units, where each unit corresponds to a letter (a-z) at one
    of four positions.

    Parameters
    ----------
    word : str
        The word to get the orthographic representation for.

    Returns
    -------
    orth_repr : ndarray
        The orthographic representation of the word.
    """
    orth_repr = np.zeros(4 * 26, dtype="float")
    for pos, letter in enumerate(word):
        orth_repr[ascii_lowercase.index(letter) + pos * 26] = 1
    return orth_repr


def get_lex_repr(word, lex, cloze_prob=1.0, cloze_multiplier=2.0):
    """Get the lexical representation for a word.

    In the Nour Eddine et al. (2023) model, there is one dedicated lexical unit for each
    word in the vocabulary. At a cloze probability of 1.0, the lexical representation is
    a one-hot encoded vector.

    Parameters
    ----------
    word : str
        The word to get the lexical representation for.
    lex : list[str]
        The lexicon of the model.
    cloze_prob : float
        The cloze probability of the word, determining how much the word is activated
        relative to the other words in the lexicon.
    cloze_multiplier : float
        How much more words with a high cloze probability are activated than words with
        a low cloze probability.

    Returns
    -------
    lex_repr : ndarray
        The lexical representation of the word.
    """
    act_word = 2 * cloze_prob
    act_other = (2 / (len(lex) - 1)) * (1 - cloze_prob)

    lex_repr = np.empty(len(lex), dtype="float")
    lex_repr.fill(act_other)
    lex_repr[lex.index(word)] = act_word
    return lex_repr


def _assemble_semantic_features(data_path, lex_size=1579):
    """Assemble sparse semantic feature vectors for each word in the lexicon.

    Parameters
    ----------
    data_path : str
        Path to the ``helper_txt_files`` folder.
    lex_size : int
        The size of the lexicon of the model (=number of lexical units).

    Returns
    -------
    sem_features : ndarray (n_sem_features, n_words)
        The semantic feature vectors.
    """
    # Read concreteness values for the first 512 words.
    is_concrete = np.loadtxt(f"{data_path}/1579words_conc_values.txt", dtype="int")
    is_concrete = np.pad(is_concrete, (0, lex_size - len(is_concrete)))

    shared_feats_block = np.hstack(
        [np.repeat(np.eye(2**i), 2 ** (9 - i), axis=0) for i in range(9, 0, -1)]
    ).T  # number of shared features is 9
    conc_feats = np.repeat(np.eye(256), 9, axis=0)
    concrete_block = np.zeros((conc_feats.shape[0], 512))
    concrete_block[:, np.flatnonzero(is_concrete)] = conc_feats
    shared_and_conc_block = np.vstack([shared_feats_block, concrete_block])

    n_filler_items = lex_size - shared_and_conc_block.shape[1]
    pad_with_zero = np.pad(shared_and_conc_block, ((0, 0), (0, n_filler_items)))
    filler_feats = np.hstack(
        [
            np.zeros((n_filler_items * 9, 512)),
            np.repeat(np.eye(n_filler_items), 9, axis=0),
        ]
    )
    return np.vstack([pad_with_zero, filler_feats])
