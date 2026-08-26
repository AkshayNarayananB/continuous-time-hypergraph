import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- FOOLPROOF PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_SYNTHETIC = os.path.join(PROJECT_ROOT, "data", "synthetic")
os.makedirs(PLOTS_DIR, exist_ok=True)
# ----------------------------

print("--- Loading Synthetic Data ---")
try:
    df_edges = pd.read_pickle(os.path.join(DATA_SYNTHETIC, "synthetic_edges.pkl"))
    df_labels = pd.read_pickle(os.path.join(DATA_SYNTHETIC, "synthetic_labels.pkl"))
except FileNotFoundError:
    print("Error: Run data_loader.py first to generate the synthetic data.")
    exit()

final_t = df_labels['time_step'].max()
drift_t = final_t // 2

print("\n--- Running Advanced Exploratory Data Analysis (EDA) ---")

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# PLOT 1: Hyperedge Cardinality Distribution
# Proves the existence of higher-order interactions
# ---------------------------------------------------------
print("Generating Plot 1: Cardinality Distribution...")
df_edges['k_size'] = df_edges['hyperedge_nodes'].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df_edges['k_size'], bins=range(2, df_edges['k_size'].max() + 2), discrete=True, color='teal')
plt.title("Hyperedge Cardinality Distribution\n(Proof of Higher-Order Group Interactions)", fontsize=14)
plt.xlabel("Number of Nodes in Hyperedge (k)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_eda_cardinality.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# PLOT 2: Community Evolution / Concept Drift Proof
# Visually demonstrates non-stationary distribution shifts
# ---------------------------------------------------------
print("Generating Plot 2: Community Evolution (Concept Drift)...")
# Group by time and cluster to get the size of each community over time
community_sizes = df_labels.groupby(['time_step', 'cluster']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 5))
community_sizes.plot(kind='area', stacked=True, alpha=0.8, colormap='Set2', ax=plt.gca())
plt.axvline(x=drift_t, color='black', linestyle='--', linewidth=2, label='Injected Concept Drift')
plt.title("Dynamic Community Evolution Over Time\n(Visualizing Non-Stationary Shift / Cluster Collapse)", fontsize=14)
plt.xlabel("Time Step")
plt.ylabel("Number of Nodes in Cluster")
plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_eda_community_drift.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# PLOT 3: Temporal Event Density
# ---------------------------------------------------------
print("Generating Plot 3: Temporal Event Density...")
temporal_density = df_edges.groupby('time_step').size()
plt.figure(figsize=(10, 4))
plt.plot(temporal_density.index, temporal_density.values, marker='o', linestyle='-', color='firebrick', linewidth=2)
plt.axvline(x=drift_t, color='k', linestyle='--', label='Injected Concept Drift')
plt.title("Hyperedge Arrival Rate over Time", fontsize=14)
plt.xlabel("Time Step")
plt.ylabel("Interaction Volume")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_eda_temporal_density.png"), dpi=300)
plt.close()
from collections import Counter

# ---------------------------------------------------------
# PLOT 4 & 5: Static Graph Projection & Topological Diagnostics
# ---------------------------------------------------------
print("Projecting Hypergraph to Static Graph for Topological EDA...")
G = nx.Graph()
num_nodes = df_labels['node_id'].nunique()
G.add_nodes_from(range(num_nodes))

# Build graph
for _, row in df_edges.iterrows():
    nodes = row['hyperedge_nodes']
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if G.has_edge(nodes[i], nodes[j]):
                G[nodes[i]][nodes[j]]['weight'] += 1
            else:
                G.add_edge(nodes[i], nodes[j], weight=1)

# Power-Law Degree Distribution (Log-Log Scatter)
print("Generating Plot 4: Degree Distribution (Power-Law Scatter)...")
degrees = [d for n, d in G.degree(weight='weight')]
degree_counts = Counter(degrees)
x_deg = list(degree_counts.keys())
y_deg = list(degree_counts.values())

plt.figure(figsize=(8, 5))
plt.scatter(x_deg, y_deg, color='purple', alpha=0.6, edgecolors='none')
plt.xscale('log')
plt.yscale('log')
plt.title("Node Degree Distribution (Log-Log Scale)\n(Checking for Scale-Free/Hub-and-Spoke Topology)", fontsize=14)
plt.xlabel("Node Degree (Log)")
plt.ylabel("Frequency (Log)")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_eda_degree_loglog.png"), dpi=300)
plt.close()

# Edge Weight Distribution (Log-Y Scatter)
print("Generating Plot 5: Edge Weight Distribution...")
weights = [d['weight'] for u, v, d in G.edges(data=True)]
weight_counts = Counter(weights)
x_wt = list(weight_counts.keys())
y_wt = list(weight_counts.values())

plt.figure(figsize=(8, 5))
plt.scatter(x_wt, y_wt, color='darkorange', alpha=0.6, edgecolors='none')
plt.yscale('log')
plt.title("Pairwise Edge Weight Distribution", fontsize=14)
plt.xlabel("Interaction Frequency (Weight)")
plt.ylabel("Count of Edge Pairs (Log Scale)")
plt.grid(True, which="major", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_eda_edge_weights.png"), dpi=300)
plt.close()

print(f"\nAdvanced EDA Complete! Check the '{PLOTS_DIR}' folder for the plots.")