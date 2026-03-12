# Word2Vec from Scratch in NumPy

A clean implementation of **Word2Vec Skip-Gram with Negative Sampling (SGNS)** in pure **NumPy**, based on the methodology from Jurafsky & Martin-style exposition:

- binary classifier for `(target, context)` pairs
- sigmoid over the dot product
- two embedding matrices:
  - `W` for **target/input embeddings**
  - `C` for **context/output embeddings**
- negative sampling
- stochastic gradient descent updates

This project is meant to be educational: the goal is not only to train embeddings, but also to make the **forward pass, loss, gradients, and parameter updates** easy to understand.

---

## Overview

Given a target word `w` and a candidate context word `c`, the model estimates:

$$
P(+|w,c) = \sigma(c \cdot w)
$$

where:

- $w$ is the target embedding vector
- $c$ is the context embedding vector
- $\sigma(x) = \frac{1}{1 + e^{-x}}$is the sigmoid

For a real context word, we want this probability to be high.  
For a randomly sampled noise word, we want it to be low.

Using **negative sampling**, for each positive pair `(w, c_pos)` we also sample `k` negative context words `c_neg1, ..., c_negk` and optimize the loss:

$$
L(w,c_{pos},c_{neg*}) =
-\log \sigma(c_{pos}\cdot w)
-\sum_{i=1}^{k}\log \sigma(-c_{neg_i}\cdot w)
$$

This pushes:

- `w` and `c_pos` **closer together**
- `w` and `c_neg` **further apart**

---

## Features

- pure NumPy implementation
- skip-gram training objective
- negative sampling with unigram distribution raised to the power `0.75`
- separate target and context embedding matrices
- explicit forward pass and SGD update
- easy to read and modify for coursework or interviews

---

## Project Structure

```text
.
├── README.md
├── word2vec.py           # main implementation
|── rural.txt      # optional demo / training script
└── requirements.txt

