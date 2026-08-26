import torch
import torch.nn as nn
import numpy as np

class TimeEncoder(nn.Module):
    """
    Projects continuous time differences (delta_t) into a Fourier feature space.
    This allows the network to understand short vs. long intervals between interactions.
    """
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        # Learnable frequencies for the Fourier transform
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                     .float().reshape(dimension, -1))
        self.w.bias = nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t: torch.Tensor):
        # t shape: [batch_size, 1]
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        output = torch.cos(self.w(t))
        return output

class NodeMemoryUpdater(nn.Module):
    """
    Maintains a continuous GRU memory state for every node in the network.
    """
    def __init__(self, num_nodes: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        
        # The actual memory bank storing the state of all nodes: [num_nodes, memory_dim]
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update_t', torch.zeros(num_nodes))
        
        self.time_encoder = TimeEncoder(time_dim)
        
        # GRU takes the concatenated (incoming message + time encoding) to update memory
        self.updater = nn.GRUCell(input_size=memory_dim + time_dim, hidden_size=memory_dim)

    def forward(self, node_indices: torch.Tensor, messages: torch.Tensor, current_t: torch.Tensor):
        """
        Updates the memory of specific nodes based on new hyperedge events.
        """
        # 1. Calculate time elapsed since last interaction
        delta_t = current_t - self.last_update_t[node_indices]
        time_features = self.time_encoder(delta_t)
        
        # 2. Prepare GRU inputs
        gru_input = torch.cat([messages, time_features], dim=-1)
        current_memory = self.memory[node_indices]
        
        # 3. Update memory
        updated_memory = self.updater(gru_input, current_memory)
        
        # 4. Write back to the memory buffer
        self.memory[node_indices] = updated_memory
        self.last_update_t[node_indices] = current_t
        
        return updated_memory
    
    def get_memory(self, node_indices: torch.Tensor = None):
        if node_indices is None:
            return self.memory
        return self.memory[node_indices]
    
    def reset_memory(self):
        self.memory.zero_()
        self.last_update_t.zero_()