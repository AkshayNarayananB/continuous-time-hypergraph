# TempHyper: Feature-Injected Continuous-Time Hypergraph Memory for Dynamic Link Prediction

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **Feature-Injected Continuous-Time Hypergraph Memory for Dynamic Link Prediction** by Akshay Narayanan Balajee Viswanath (Nanyang Technological University, Singapore).

## 📖 Abstract
To properly model real communication networks, we need to capture how they change over time and how multiple people interact at once. Standard dynamic graph models force these group interactions into pairs, which loses group context and slows down computation. Static hypergraph models, conversely, keep the group structure intact but miss how relationships evolve. 

**TempHyper** is a neural network that combines a Gated Recurrent Unit (GRU) with Hypergraph Attention (HyperGAT). It introduces a **feature injection module** to fuse past interaction counts with deep network embeddings, allowing the model to accurately predict both repeated and entirely new group interactions. 

## ✨ Key Contributions
- **Continuous-Time Memory:** A memory-augmented hypergraph attention framework processing multi-party interactions natively without pairwise reduction.
- **Feature-Injected Prediction Head:** Separates historical memorization from true structural forecasting, allowing for strong performance on entirely novel interactions.
- **Topological Semantic Mismatch:** Mathematical and empirical demonstration of why static organizational node labels fail in clustering approaches on shifting networks.

## ⚙️ Setup & Installation

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage your environment. The environment is configured for standard PyTorch and PyTorch Geometric workflows.

```bash
# 1. Clone the repository
git clone https://github.com/AkshayNarayananB/continuous-time-hypergraph.git
cd continuous-time-hypergraph

# 2. Create and activate a new conda environment
conda create -n continuous-time-hypergraph python=3.10 -y
conda activate continuous-time-hypergraph

# 3. Install PyTorch (Adjust the CUDA version based on your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install PyTorch Geometric and additional dependencies
pip install torch_geometric
pip install -r requirements.txt
```

## 🚀 Usage 

*Provide a brief example of how to execute the training script. Replace with your actual script names.*

```bash
# Train TempHyper on the Stanford Email dataset
python train.py --dataset email --epochs 50 --batch_size 128

# Evaluate the model on novel interactions
python evaluate.py --dataset email --split novel
```

## 📊 Key Results

**Link Prediction Performance (ROC-AUC)**

| Architecture | Stanford Email (ALL) | Stanford Email (NOVEL) | MathOverflow (ALL) | MathOverflow (NOVEL) |
| :--- | :---: | :---: | :---: | :---: |
| Freq. Heuristic | 0.8276 | 0.7084 | 0.6161 | 0.5992 |
| Static GAT | 0.7862 | 0.7669 | 0.6251 | 0.6325 |
| Static HGNN | 0.7860 | 0.7505 | 0.6660 | 0.6607 |
| CT RNN | 0.8680 | 0.8552 | 0.5713 | 0.5685 |
| **TempHyper (Ours)** | **0.9004** | **0.8674** | **0.6947** | **0.6348** |

**Computational Runtime Benchmark**

| Dataset | TempHyper (O(N)) | CT-RNN (O(N²))
| :--- | :---: | :---: |
| Email | 0.94 ms/step | 6.30 ms/step |
| MathOverflow | 1.30 ms/step | 3.07 ms/step |

*Note: Natively preserving hypergraph structures introduces minor constant-time control-flow overhead due to sparse index-gathering. This trades execution speed for the complete preservation of higher-order structural context.*

## 📝 Citation
If you find this code or our paper useful for your research, please cite:
```bibtex
@article{viswanath2026temphyper,
  title={Feature-Injected Continuous-Time Hypergraph Memory for Dynamic Link Prediction},
  author={Viswanath, Akshay Narayanan Balajee},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
