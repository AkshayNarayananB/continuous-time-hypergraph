import torch
import torch.nn.functional as F

def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Computes the target distribution P from the soft assignments Q.
    This forces the network to make high-confidence clustering decisions,
    pushing embeddings closer to their assigned cluster centers.
    
    Formula: p_{ij} = (q_{ij}^2 / f_j) / sum_k (q_{ik}^2 / f_k)
    where f_j = sum_i q_{ij} (soft cluster frequency)
    """
    weight = (q ** 2) / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p

def dec_kl_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Standard Kullback-Leibler Divergence loss for Deep Embedding Clustering.
    Minimizing this aligns the predicted soft-assignments Q with the strict targets P.
    """
    # PyTorch's kl_div expects log probabilities for the input and standard probs for the target
    return F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')

def temporal_drift_penalty(current_centroids: torch.Tensor, prev_centroids: torch.Tensor) -> torch.Tensor:
    """
    The Non-Stationary Concept Drift Penalty.
    Penalizes abrupt jumps in cluster centroids between time step t and t-1.
    Prevents catastrophic cluster collapse when the temporal topology suddenly shifts.
    """
    if prev_centroids is None:
        return torch.tensor(0.0, device=current_centroids.device, requires_grad=True)
    
    # We use Mean Squared Error as an efficient proxy for Wasserstein distance 
    # to maintain smooth centroid migration trajectories.
    return F.mse_loss(current_centroids, prev_centroids)