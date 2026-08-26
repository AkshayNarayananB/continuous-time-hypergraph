import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC

def generate_missing_pdf(dataset="email"):
    print(f"--- Forcing t-SNE PDF Generation for {dataset.upper()} ---")
    
    # 1. Load Data
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    df_edges = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_edges.pkl"))
    df_labels = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_labels.pkl"))

    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted(all_unique_nodes))}
    df_labels['node_id'] = df_labels['node_id'].map(node_mapping)
    num_nodes = len(node_mapping)

    # 2. Load the Saved Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TempHyperDEC(num_nodes, memory_dim=64, time_dim=32, hidden_dim=32, n_clusters=2, use_attention=True).to(device)
    
    model_path = os.path.join(PROJECT_ROOT, "temphyper_link_pred.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Cannot find {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    print("-> Successfully loaded saved model weights.")

    # 3. Extract Embeddings
    z_active = (model.memory_module.get_memory() + model.node_emb.weight).detach().cpu().numpy()
    
    # 4. Bulletproof Label Extraction
    node_to_cluster = df_labels.groupby('node_id')['cluster'].last().to_dict()
    
    valid_indices = []
    valid_labels = []
    
    for i in range(z_active.shape[0]):
        # Check if the node has a label AND isn't NaN
        if i in node_to_cluster and pd.notna(node_to_cluster[i]):
            valid_indices.append(i)
            valid_labels.append(str(node_to_cluster[i])) # Force to string for seaborn
            
    z_filtered = z_active[valid_indices]
    labels_filtered = np.array(valid_labels)
    
    # Remove NaNs from the embeddings themselves
    nan_mask = ~np.isnan(z_filtered).any(axis=1)
    z_filtered = z_filtered[nan_mask]
    labels_filtered = labels_filtered[nan_mask]
    
    print(f"-> Found {z_filtered.shape[0]} valid nodes with ground-truth labels.")

    # Filter to top 7 departments so the legend isn't massive
    top_departments = pd.Series(labels_filtered).value_counts().nlargest(7).index.tolist()
    mask = np.isin(labels_filtered, top_departments)
    z_final = z_filtered[mask]
    labels_final = labels_filtered[mask]

    # 5. Generate and Save PDF
    if z_final.shape[0] > 10:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
        plt.figure(figsize=(8, 8))
        
        print("-> Running TSNE calculations...")
        tsne = TSNE(n_components=2, perplexity=min(30, z_final.shape[0]-1), random_state=42, init='pca', learning_rate='auto')
        z_2d = tsne.fit_transform(z_final)
        
        print("-> Drawing scatter plot...")
        sns.scatterplot(
            x=z_2d[:, 0], y=z_2d[:, 1], 
            hue=labels_final, 
            palette="Set2", 
            s=100, alpha=0.85, edgecolor='w', linewidth=0.5
        )
        
        plt.title("Topological Semantic Mismatch (Latent Space)", fontweight='bold', pad=15)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title="Ground Truth Dept", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig2_path = os.path.join(PROJECT_ROOT, "tsne_latent_space.pdf")
        plt.savefig(fig2_path, format='pdf', bbox_inches='tight')
        print(f"\n[SUCCESS] Saved missing PDF to: {fig2_path}")
    else:
        print("-> Error: Not enough valid ground-truth nodes to plot.")

if __name__ == "__main__":
    generate_missing_pdf()