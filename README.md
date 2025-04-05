# Character Prefix Conditioning

## Setup

When using a language model for code completion, we typically want the model to produce a completion that begins with what the user has typed.

However, modern language models operate on sequences of tokens, not characters, so naively tokenizing the user's input and sending it to the model produces wrong results if the user's cursor doesn't happen to lie on a token boundary.

Instead, we need an algorithm that samples a sequence of tokens conditional on a prefix of characters, rather than the more typical case of sampling conditional on a prefix of tokens.

We call this **character prefix conditioning**, an algorithm for sampling a sequence of tokens conditioned on a character prefix.

We want to sample a sequence of tokens $s = t_1, t_2, ... , t_n$ from a distribution specified by an autoregressive model $p(s)$ given by

$$p(s) = p(t_1, t_2, ... , t_n) = \prod_{k=1}^{n} p(t_k|t_1, ... , t_{k-1})$$

subject to the constraint that $s$ starts with a character prefix $P$, i.e. $P$ is a prefix of $\text{repr}(t_1) + \text{repr}(t_2) + ... + \text{repr}(t_n)$, where $+$ means string concatenation and $\text{repr}$ maps a token to the characters it represents.

We define $q(s) = p(s \mid s \text{ starts with } P)$. It's sufficient to find a way to sample autoregressively from $q(s)$, that is, to sample from $q(t_k|t_1, ... , t_{k-1})$ for each $k$.

## Problem

Can you construct an efficient algorithm for sampling from $q(t_k|t_1, ... , t_{k-1})$, that minimizes calls to the original language model?

## Solution
This problem is actually trickier than it initially seems, and the following examples explain why:
- [Example 1](https://x.com/jbfja/status/1876492234117505175)
- [Example 2](https://news.ycombinator.com/item?id=42670128)

The solution is to use a top-down dynamic programming approach with a partition function $Z(\text{pos}, \text{leftover\\_str})$ that computes the total probability mass of all valid ways to complete the prefix from a given position ${\text{pos}}$. So, when sampling a token, we do:

$$p(t \mid P) \propto p(t) \cdot Z(\text{pos}, \text{leftover\\_str})$$

This means we never need to branch out and calculate probabilities for every possible tokenization path since we summarize all future paths via the total probability mass of all valid ways to complete the remaining prefix from a given position for each state $(\text{pos}, \text{leftover\\_str})$
and cache it. This does not blow up in terms of model calls and scales linearly with the prefix length while still sampling from $q(s)$.
