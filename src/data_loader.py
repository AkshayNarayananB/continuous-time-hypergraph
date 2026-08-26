import os
import urllib.request
import gzip
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
import logging
from typing import Tuple

# --- FOOLPROOF PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_SYNTHETIC = os.path.join(PROJECT_ROOT, "data", "synthetic")
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
# ----------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticDynamicHypergraph:
    """
    [Keeping the synthetic generator exactly as it was]
    """
    def __init__(self, num_nodes: int = 1500, num_time_steps: int = 50, num_clusters: int = 5, save_dir: str = DATA_SYNTHETIC):
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.num_clusters = num_clusters
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def generate(self, base_hyperedges: int = 400, max_k: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        edges_path = os.path.join(self.save_dir, "synthetic_edges.pkl")
        labels_path = os.path.join(self.save_dir, "synthetic_labels.pkl")
        
        if os.path.exists(edges_path) and os.path.exists(labels_path):
            logging.info("Synthetic data already exists. Skipping generation.")
            return pd.read_pickle(edges_path), pd.read_pickle(labels_path)

        logging.info(f"Generating REALISTIC synthetic hypergraph: {self.num_nodes} nodes, {self.num_time_steps} time steps.")
        node_activity_weights = np.random.pareto(a=2.0, size=self.num_nodes) + 1.0
        node_labels = np.random.randint(0, self.num_clusters, size=self.num_nodes)
        
        hyperedges, labels_over_time = [], []
        drift_start = self.num_time_steps // 2 - 5
        drift_end = self.num_time_steps // 2 + 5

        for t in range(self.num_time_steps):
            if drift_start <= t <= drift_end:
                migration_prob = (t - drift_start) / max(1, (drift_end - drift_start))
                cluster_1_nodes = np.where(node_labels == 1)[0]
                migrate_mask = np.random.rand(len(cluster_1_nodes)) < migration_prob
                node_labels[cluster_1_nodes[migrate_mask]] = 0

            labels_over_time.append(pd.DataFrame({'node_id': np.arange(self.num_nodes), 'cluster': node_labels.copy(), 'time_step': t}))

            current_volume = int(base_hyperedges + (base_hyperedges * 0.6) * np.sin(2 * np.pi * t / 15))
            current_volume = max(50, int(np.random.normal(current_volume, current_volume * 0.1)))

            for _ in range(current_volume):
                k = min(max_k, max(2, np.random.zipf(a=2.5)))
                if np.random.rand() < 0.85:
                    chosen_cluster = np.random.choice(np.unique(node_labels))
                    candidate_nodes = np.where(node_labels == chosen_cluster)[0]
                    if len(candidate_nodes) >= k:
                        probs = node_activity_weights[candidate_nodes] / node_activity_weights[candidate_nodes].sum()
                        edge_nodes = np.random.choice(candidate_nodes, size=k, replace=False, p=probs)
                    else:
                        probs = node_activity_weights / node_activity_weights.sum()
                        edge_nodes = np.random.choice(np.arange(self.num_nodes), size=k, replace=False, p=probs)
                else:
                    probs = node_activity_weights / node_activity_weights.sum()
                    edge_nodes = np.random.choice(np.arange(self.num_nodes), size=k, replace=False, p=probs)
                hyperedges.append({'time_step': t, 'hyperedge_nodes': tuple(sorted(edge_nodes))})

        df_hyperedges = pd.DataFrame(hyperedges)
        df_labels = pd.concat(labels_over_time, ignore_index=True)
        
        df_hyperedges.to_pickle(edges_path)
        df_labels.to_pickle(labels_path)
        return df_hyperedges, df_labels

class RealDatasetDownloader:
    def __init__(self, data_dir: str = DATA_RAW):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _download_file(self, url: str, filename: str) -> str:
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            logging.info(f"Downloading dataset from {url}...")
            urllib.request.urlretrieve(url, file_path)
            logging.info("Download complete.")
        return file_path

    def process_snap_email(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        edges_out = os.path.join(self.data_dir, "email_edges.pkl")
        labels_out = os.path.join(self.data_dir, "email_labels.pkl")
        
        if os.path.exists(edges_out) and os.path.exists(labels_out):
            logging.info("SNAP Email data already processed. Loading from disk.")
            return pd.read_pickle(edges_out), pd.read_pickle(labels_out)

        # 1. Download Interactions and Labels
        edges_file = self._download_file("http://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz", "email-Eu-core-temporal.txt.gz")
        labels_file = self._download_file("http://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz", "email-Eu-core-department-labels.txt.gz")
        
        logging.info("Processing SNAP Email datasets...")
        
        # 2. Process Labels (Static Ground Truth)
        df_static_labels = pd.read_csv(labels_file, sep=' ', header=None, names=['node_id', 'cluster'])
        
        # 3. Process Edges (Windowed by Day = 86400 seconds)
        df_edges = pd.read_csv(edges_file, sep=' ', header=None, names=['source', 'target', 'timestamp'])
        df_edges = df_edges.sort_values(by='timestamp')
        df_edges['time_step'] = df_edges['timestamp'] // 86400
        
        # Aggregate hyperedges
        hyperedges = df_edges.groupby(['time_step', 'source'])['target'].apply(lambda x: list(set(x))).reset_index()
        hyperedges['hyperedge_nodes'] = hyperedges.apply(lambda row: tuple(sorted([row['source']] + row['target'])), axis=1)
        hyperedges['k_size'] = hyperedges['hyperedge_nodes'].apply(len)
        df_final_edges = hyperedges[hyperedges['k_size'] >= 2][['time_step', 'hyperedge_nodes']]
        
        # 4. Broadcast Static Labels across active time steps to match pipeline architecture
        time_steps = sorted(df_final_edges['time_step'].unique())
        labels_over_time = []
        for t in time_steps:
            df_t = df_static_labels.copy()
            df_t['time_step'] = t
            labels_over_time.append(df_t)
            
        df_final_labels = pd.concat(labels_over_time, ignore_index=True)

        df_final_edges.to_pickle(edges_out)
        df_final_labels.to_pickle(labels_out)
        logging.info(f"Saved SNAP Email: {len(df_final_edges)} edges across {len(time_steps)} days.")
        return df_final_edges, df_final_labels

    def process_mathoverflow(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        edges_out = os.path.join(self.data_dir, "math_edges.pkl")
        labels_out = os.path.join(self.data_dir, "math_labels.pkl")
        
        if os.path.exists(edges_out) and os.path.exists(labels_out):
            logging.info("MathOverflow data already processed. Loading from disk.")
            return pd.read_pickle(edges_out), pd.read_pickle(labels_out)

        edges_file = self._download_file("http://snap.stanford.edu/data/sx-mathoverflow.txt.gz", "sx-mathoverflow.txt.gz")
        
        logging.info("Processing MathOverflow edges (Windowed by Week)...")
        df_edges = pd.read_csv(edges_file, sep=' ', header=None, names=['source', 'target', 'timestamp'])
        df_edges = df_edges.sort_values(by='timestamp')
        
        # Window by Week (604800 seconds) to compress sparsity
        df_edges['time_step'] = df_edges['timestamp'] // 604800
        
        # Aggregate hyperedges
        hyperedges = df_edges.groupby(['time_step', 'target'])['source'].apply(lambda x: list(set(x))).reset_index()
        hyperedges['hyperedge_nodes'] = hyperedges.apply(lambda row: tuple(sorted([row['target']] + row['source'])), axis=1)
        hyperedges['k_size'] = hyperedges['hyperedge_nodes'].apply(len)
        df_final_edges = hyperedges[hyperedges['k_size'] >= 2][['time_step', 'hyperedge_nodes']]
        
        # --- GENERATING PSEUDO-LABELS FOR BENCHMARKING ---
        logging.info("MathOverflow has no official labels. Generating structural Pseudo-Labels via Louvain...")
        G = nx.Graph()
        for _, row in df_edges.iterrows():
            if G.has_edge(row['source'], row['target']):
                G[row['source']][row['target']]['weight'] += 1
            else:
                G.add_edge(row['source'], row['target'], weight=1)
                
        louvain_partition = louvain_communities(G, weight='weight')
        node_to_cluster = {}
        for cluster_id, community in enumerate(louvain_partition):
            for node in community:
                node_to_cluster[node] = cluster_id
                
        df_static_labels = pd.DataFrame(list(node_to_cluster.items()), columns=['node_id', 'cluster'])
        
        # Broadcast across time steps
        time_steps = sorted(df_final_edges['time_step'].unique())
        labels_over_time = []
        for t in time_steps:
            df_t = df_static_labels.copy()
            df_t['time_step'] = t
            labels_over_time.append(df_t)
            
        df_final_labels = pd.concat(labels_over_time, ignore_index=True)

        df_final_edges.to_pickle(edges_out)
        df_final_labels.to_pickle(labels_out)
        logging.info(f"Saved MathOverflow: {len(df_final_edges)} edges across {len(time_steps)} weeks.")
        return df_final_edges, df_final_labels


if __name__ == "__main__":
    print("\n--- Generating Real Dataset 1 (SNAP Email) ---")
    downloader = RealDatasetDownloader()
    df_email_edges, df_email_labels = downloader.process_snap_email()
    print("Sample Email Labels (Ground Truth Departments):\n", df_email_labels.head())

    print("\n--- Generating Real Dataset 2 (MathOverflow) ---")
    df_math_edges, df_math_labels = downloader.process_mathoverflow()
    print("Sample MathOverflow Pseudo-Labels (Louvain Communities):\n", df_math_labels.head())