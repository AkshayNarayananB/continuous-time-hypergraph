# TempHyper: Feature-Injected Continuous-Time Hypergraph Memory for Dynamic Link Prediction

[![arXiv](https://img.shields.io/badge/arXiv-26xx.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Official PyTorch implementation of the paper: **"Feature-Injected Continuous-Time Hypergraph Memory for Dynamic Link Prediction"** by Akshay Narayanan Balajee Viswanath (Nanyang Technological University, Singapore).

---

## 📖 Abstract

Real-world communication and interaction networks inherently comprise concurrent multi-party interactions that cannot be faithfully reduced to independent dyadic pairs. Standard dynamic graph models decompose group interactions into pairwise cliques, incurring a severe $\mathcal{O}(N^2)$ computational complexity explosion and destroying latent multi-way structural context. Conversely, static hypergraph models preserve structural integrity but fail to track continuous-time evolution.

To bridge this gap, we introduce **TempHyper** (MARUT), an end-to-end continuous-time framework combining **Gated Recurrent Unit (GRU) node memory** with **two-stage Hypergraph Attention (HyperGAT)**. Furthermore, to mitigate the widespread memorization trap where models overfit to historical interaction frequencies rather than true structural topologies, TempHyper incorporates a **feature-injection prediction head** that explicitly decouples frequency heuristics from deep network embeddings.

---

## ✨ Key Contributions

- **Native Dynamic Hypergraph Modeling:** Interleaves continuous-time recurrent node memory with HyperGAT, efficiently processing multi-party interactions without pairwise expansion.
- **Decoupled Inductive Link Prediction:** Introduces a feature-injected prediction architecture that markedly improves forecasting on novel, unobserved groups.
- **Topological Semantic Mismatch Analysis:** Formally identifies and empirically demonstrates why static ground-truth organizational labels fail to correlate with fluid, time-evolving communication network topologies.

---

## 🗂️ Repository Structure

```text
├── data/                  # Dataset loaders (Stanford Email, MathOverflow)
├── models/
│   ├── memory.py          # Continuous-time GRU node memory modules
│   ├── hypergat.py        # Two-stage node-to-hyperedge attention layers
│   └── temphyper.py       # Full TempHyper architecture & prediction head
├── utils/                 # Metrics, streaming batch protocols, and helpers
├── train.py               # Main training script
├── evaluate.py            # Streaming evaluation & novel link forecasting
└── requirements.txt       # Project dependencies
